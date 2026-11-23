#include "veyra/pipeline/VfxSrBackend.h"
#include "veyra/pipeline/VfgBackend.h"
#include "veyra/engine/EnhancementSettings.h"
#include "veyra/gfx/D3D12DeviceContext.h"
#include "veyra/Log.h"
#include <Windows.h>
#include <wrl/client.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <format>
#include <mutex>
#include <thread>
#include <vector>
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
#include <cuda.h>
#endif

namespace veyra::pipeline {
namespace {
namespace fs=std::filesystem;
constexpr DWORD kDllFlags=LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32|LOAD_LIBRARY_SEARCH_USER_DIRS;
bool regularFile(const fs::path& path){std::error_code error;return fs::is_regular_file(path,error);}
HMODULE load(const fs::path& path){
    const auto dll=LoadLibraryExW(path.c_str(),nullptr,kDllFlags);
    if(!dll){const auto bytes=path.u8string();
        log::error("vfx-sr",std::format("LoadLibrary failed path={} win32={}",std::string(reinterpret_cast<const char*>(bytes.data()),bytes.size()),GetLastError()));}
    return dll;
}
template<class T> bool bind(HMODULE dll,const char* name,T& fn){
    fn=reinterpret_cast<T>(GetProcAddress(dll,name));
    if(!fn)log::error("vfx-sr",std::format("missing API {} win32={}",name,GetLastError()));
    return fn!=nullptr;
}
bool hr(HRESULT value,const char* operation){
    if(SUCCEEDED(value))return true;
    log::error("vfx-sr",std::format("{} HRESULT=0x{:08X}",operation,unsigned(value)));return false;
}
}

bool VfxSrBackend::runtimeAvailable(const std::wstring& root){
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    const fs::path directory=VfgBackend::runtimeDirectory(root);
    if(directory.empty())return false;
    for(auto name:{L"NVVideoEffects.dll",L"NVCVImage.dll",L"nvngxruntime.dll",L"cudart64_12.dll",L"nvVFXVideoSuperRes.dll",L"nvngx_vsr.dll"})
        if(!regularFile(directory/name))return false;
    return true;
#else
    (void)root;return false;
#endif
}

struct VfxSrBackend::Impl {
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    int (__cdecl* create)(const char*,void**)=nullptr;
    void (__cdecl* destroy)(void*)=nullptr;
    int (__cdecl* setU32)(void*,const char*,unsigned)=nullptr;
    int (__cdecl* setImage)(void*,const char*,void*)=nullptr;
    int (__cdecl* setStream)(void*,const char*,CUstream)=nullptr;
    int (__cdecl* loadEffect)(void*)=nullptr;
    int (__cdecl* runEffect)(void*,int)=nullptr;
    int (__cdecl* imageCreate)(unsigned,unsigned,int,int,unsigned,unsigned,unsigned,void**)=nullptr;
    void (__cdecl* imageDealloc)(void*)=nullptr;
    void (__cdecl* imageDestroy)(void*)=nullptr;
    int (__cdecl* imageInit)(void*,unsigned,unsigned,int,void*,int,int,unsigned,unsigned)=nullptr;
    int (__cdecl* fromD3D)(DXGI_FORMAT,int*,int*,unsigned char*)=nullptr;
    const char* (__cdecl* errorString)(int)=nullptr;
#define CU_FN(name) decltype(&name) name##Fn=nullptr
    CU_FN(cuInit);CU_FN(cuDeviceGetCount);CU_FN(cuDeviceGet);CU_FN(cuDeviceGetLuid);CU_FN(cuDeviceGetAttribute);
    CU_FN(cuDevicePrimaryCtxRetain);CU_FN(cuDevicePrimaryCtxRelease);CU_FN(cuCtxPushCurrent);CU_FN(cuCtxPopCurrent);
    CU_FN(cuStreamCreate);CU_FN(cuStreamDestroy);CU_FN(cuStreamSynchronize);CU_FN(cuStreamQuery);
    CU_FN(cuImportExternalMemory);CU_FN(cuExternalMemoryGetMappedBuffer);CU_FN(cuDestroyExternalMemory);CU_FN(cuMemFree);
    CU_FN(cuImportExternalSemaphore);CU_FN(cuDestroyExternalSemaphore);CU_FN(cuWaitExternalSemaphoresAsync);CU_FN(cuSignalExternalSemaphoresAsync);
    CU_FN(cuGetErrorName);CU_FN(cuGetErrorString);
#undef CU_FN
    struct Buffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        CUexternalMemory memory=nullptr;
        CUdeviceptr pointer=0;
        void* image=nullptr;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        uint64_t bytes=0;
    };
    Buffer input,output;
    gfx::D3D12DeviceContext* context=nullptr;
    Microsoft::WRL::ComPtr<ID3D12Fence> producer,completion;
    CUexternalSemaphore inputSemaphore=nullptr,outputSemaphore=nullptr;
    CUcontext cudaContext=nullptr;CUstream stream=nullptr;CUdevice device=0;
    void* effect=nullptr;
    static constexpr unsigned kMaxPending=4;
    bool retained=false,ready=false,asynchronous=true;
    uint64_t signalValue=0,queuedFence=0;
    std::mutex jobsMutex,cudaSubmissionMutex;
    std::condition_variable jobsChanged;
    std::deque<uint64_t> jobs;
    std::thread worker;
    bool stopping=false;
    std::atomic<bool> failed{false};
    unsigned outstanding=0;
    std::vector<HMODULE> libraries;
    bool cv(int code,const char* name){
        if(code==0)return true;
        log::error("vfx-sr",std::format("{} status={} detail={}",name,code,errorString?errorString(code):"unavailable"));return false;
    }
    bool cu(CUresult code,const char* name){
        if(code==CUDA_SUCCESS)return true;
        const char* label=nullptr;const char* detail=nullptr;
        if(cuGetErrorNameFn)cuGetErrorNameFn(code,&label);
        if(cuGetErrorStringFn)cuGetErrorStringFn(code,&detail);
        log::error("vfx-sr-cuda",std::format("{} result={} name={} detail={}",name,unsigned(code),label?label:"?",detail?detail:"?"));return false;
    }
    struct Current {
        Impl& owner;bool active=false;
        explicit Current(Impl& p):owner(p){active=p.cu(p.cuCtxPushCurrentFn(p.cudaContext),"cuCtxPushCurrent");}
        ~Current(){if(active){CUcontext previous=nullptr;owner.cu(owner.cuCtxPopCurrentFn(&previous),"cuCtxPopCurrent");}}
    };
    bool importFence(ID3D12Fence* fence,CUexternalSemaphore& semaphore){
        HANDLE handle=nullptr;
        if(!hr(context->device()->CreateSharedHandle(fence,nullptr,GENERIC_ALL,nullptr,&handle),"CreateSharedHandle fence"))return false;
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};desc.type=CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;desc.handle.win32.handle=handle;
        const auto code=cuImportExternalSemaphoreFn(&semaphore,&desc);CloseHandle(handle);
        return cu(code,"cuImportExternalSemaphore");
    }
    // A shared row-major buffer laid out as the texture's copyable footprint,
    // mapped into CUDA and wrapped (not owned) by an NvCVImage descriptor.
    bool makeBuffer(Buffer& buffer,unsigned width,unsigned height,int pixelFormat,int componentType,unsigned layout){
        D3D12_RESOURCE_DESC texture{};texture.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;texture.Width=width;texture.Height=height;
        texture.DepthOrArraySize=1;texture.MipLevels=1;texture.Format=DXGI_FORMAT_R8G8B8A8_UNORM;texture.SampleDesc.Count=1;
        UINT rows=0;uint64_t rowBytes=0,total=0;context->device()->GetCopyableFootprints(&texture,0,1,0,&buffer.footprint,&rows,&rowBytes,&total);
        buffer.bytes=uint64_t(buffer.footprint.Footprint.RowPitch)*height;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;desc.Width=buffer.bytes;desc.Height=1;
        desc.DepthOrArraySize=1;desc.MipLevels=1;desc.SampleDesc.Count=1;desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if(!hr(context->device()->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&buffer.resource)),"Create shared buffer"))return false;
        HANDLE handle=nullptr;
        if(!hr(context->device()->CreateSharedHandle(buffer.resource.Get(),nullptr,GENERIC_ALL,nullptr,&handle),"CreateSharedHandle buffer"))return false;
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC imported{};
        imported.type=CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;imported.handle.win32.handle=handle;
        imported.size=context->device()->GetResourceAllocationInfo(0,1,&desc).SizeInBytes;imported.flags=CUDA_EXTERNAL_MEMORY_DEDICATED;
        const auto code=cuImportExternalMemoryFn(&buffer.memory,&imported);CloseHandle(handle);
        if(!cu(code,"cuImportExternalMemory"))return false;
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapped{};mapped.size=buffer.bytes;
        if(!cu(cuExternalMemoryGetMappedBufferFn(&buffer.pointer,buffer.memory,&mapped),"cuExternalMemoryGetMappedBuffer"))return false;
        if(!cv(imageCreate(1,1,pixelFormat,componentType,layout,0,1,&buffer.image),"NvCVImage_Create"))return false;
        imageDealloc(buffer.image);
        return cv(imageInit(buffer.image,width,height,int(buffer.footprint.Footprint.RowPitch),reinterpret_cast<void*>(buffer.pointer),pixelFormat,componentType,layout,1),"NvCVImage_Init");
    }
    void releaseBuffer(Buffer& buffer){
        if(buffer.image){imageDestroy(buffer.image);buffer.image=nullptr;}
        if(buffer.pointer){cu(cuMemFreeFn(buffer.pointer),"cuMemFree external mapping");buffer.pointer=0;}
        if(buffer.memory){cu(cuDestroyExternalMemoryFn(buffer.memory),"cuDestroyExternalMemory");buffer.memory=nullptr;}
        buffer.resource.Reset();
    }
    bool copy(ID3D12GraphicsCommandList* list,ID3D12Resource* texture,const Buffer& buffer,bool toBuffer){
        if(!ready||failed.load()||!list||!texture)return false;
        D3D12_TEXTURE_COPY_LOCATION shared{};shared.pResource=buffer.resource.Get();shared.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;shared.PlacedFootprint=buffer.footprint;
        D3D12_TEXTURE_COPY_LOCATION image{};image.pResource=texture;image.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;image.SubresourceIndex=0;
        if(toBuffer)list->CopyTextureRegion(&shared,0,0,0,&image,nullptr);else list->CopyTextureRegion(&image,0,0,0,&shared,nullptr);
        return true;
    }
    bool cudaJob(uint64_t value){
        std::lock_guard submit(cudaSubmissionMutex);
        if(failed.load())return false;
        Current current(*this);
        if(!current.active)return false;
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait{};wait.params.fence.value=value;
        const bool waited=cu(cuWaitExternalSemaphoresAsyncFn(&inputSemaphore,&wait,1,stream),"cuWaitExternalSemaphoresAsync");
        bool ok=waited&&cv(runEffect(effect,1),"NvVFX_Run(VideoSuperRes)");
        // Publish rejection before releasing the consumer so future submissions stop accepting work.
        if(!ok)failed.store(true);
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal{};signal.params.fence.value=value;
        if(!waited||!cu(cuSignalExternalSemaphoresAsyncFn(&outputSemaphore,&signal,1,stream),"cuSignalExternalSemaphoresAsync")){
            failed.store(true);
            // A rejected or unsignaled job must never leave the graph queue waiting.
            if(waited)cu(cuStreamSynchronizeFn(stream),"rejected VideoSuperRes drain");
            hr(completion->Signal(value),"release failed VideoSuperRes job");
            return false;
        }
        return ok;
    }
    void checkCudaCompletion(){
        std::lock_guard submit(cudaSubmissionMutex);
        Current current(*this);
        const auto result=current.active?cuStreamQueryFn(stream):CUDA_ERROR_INVALID_CONTEXT;
        if(result==CUDA_SUCCESS||result==CUDA_ERROR_NOT_READY)return;
        cu(result,"asynchronous VideoSuperRes completion");
        std::lock_guard lock(jobsMutex);
        failed.store(true);
        if(queuedFence&&completion->GetCompletedValue()<queuedFence)
            hr(completion->Signal(queuedFence),"release VideoSuperRes waits after CUDA fault");
    }
    void work(){
        for(;;){
            uint64_t value=0;
            {
                std::unique_lock lock(jobsMutex);
                if(jobs.empty()&&!stopping){
                    if(!failed.load()&&queuedFence&&completion->GetCompletedValue()<queuedFence){
                        jobsChanged.wait_for(lock,std::chrono::milliseconds(4),[&]{return stopping||!jobs.empty();});
                    }else{
                        jobsChanged.wait(lock,[&]{return stopping||!jobs.empty()||
                            (!failed.load()&&queuedFence&&completion->GetCompletedValue()<queuedFence);});
                    }
                }
                if(jobs.empty()){
                    if(stopping)return;
                    const bool pending=!failed.load()&&queuedFence&&completion->GetCompletedValue()<queuedFence;
                    lock.unlock();
                    if(pending)checkCudaCompletion();
                    continue;
                }
                value=jobs.front();jobs.pop_front();
            }
            bool ok=false;
            try{ok=cudaJob(value);}catch(...){log::error("vfx-sr","SDK worker exception; disabling VideoSuperRes");}
            if(!ok){
                failed.store(true);
                std::lock_guard submit(cudaSubmissionMutex);
                Current current(*this);
                if(current.active&&stream)cu(cuStreamSynchronizeFn(stream),"rejected VideoSuperRes drain");
                if(completion->GetCompletedValue()<value)
                    hr(completion->Signal(value),"release failed VideoSuperRes job");
            }
            {std::lock_guard lock(jobsMutex);--outstanding;}
        }
    }
    void stopWorker(){
        {std::lock_guard lock(jobsMutex);stopping=true;}
        jobsChanged.notify_one();
        if(worker.joinable())worker.join();
    }
#endif
};

VfxSrBackend::VfxSrBackend():impl_(std::make_unique<Impl>()){}
VfxSrBackend::~VfxSrBackend(){shutdown();}

bool VfxSrBackend::initialize(gfx::D3D12DeviceContext& context,const std::wstring& root,unsigned inputWidth,unsigned inputHeight,
                              unsigned outputWidth,unsigned outputHeight,unsigned quality){
    shutdown();
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    auto& p=*impl_;
    if(context.adapter().vendorId!=0x10DE||!runtimeAvailable(root)||!engine::vfxVideoSr(quality)||
       !inputWidth||!inputHeight||outputWidth<inputWidth||outputHeight<inputHeight){
        log::error("vfx-sr","unsupported adapter/runtime/mode/dimensions");return false;
    }
    p.context=&context;
    std::array<wchar_t,32768> system{};const auto n=GetSystemDirectoryW(system.data(),UINT(system.size()));if(!n||n>=system.size())return false;
    HMODULE driver=load(fs::path(system.data())/L"nvcuda.dll");if(!driver)return false;p.libraries.push_back(driver);
#define BIND_CU(name,exported) if(!bind(driver,exported,p.name##Fn))return false
    BIND_CU(cuInit,"cuInit");BIND_CU(cuDeviceGetCount,"cuDeviceGetCount");BIND_CU(cuDeviceGet,"cuDeviceGet");BIND_CU(cuDeviceGetLuid,"cuDeviceGetLuid");
    BIND_CU(cuDeviceGetAttribute,"cuDeviceGetAttribute");BIND_CU(cuDevicePrimaryCtxRetain,"cuDevicePrimaryCtxRetain");
    BIND_CU(cuDevicePrimaryCtxRelease,"cuDevicePrimaryCtxRelease_v2");BIND_CU(cuCtxPushCurrent,"cuCtxPushCurrent_v2");BIND_CU(cuCtxPopCurrent,"cuCtxPopCurrent_v2");
    BIND_CU(cuStreamCreate,"cuStreamCreate");BIND_CU(cuStreamDestroy,"cuStreamDestroy_v2");BIND_CU(cuStreamSynchronize,"cuStreamSynchronize");BIND_CU(cuStreamQuery,"cuStreamQuery");
    BIND_CU(cuImportExternalMemory,"cuImportExternalMemory");BIND_CU(cuExternalMemoryGetMappedBuffer,"cuExternalMemoryGetMappedBuffer");
    BIND_CU(cuDestroyExternalMemory,"cuDestroyExternalMemory");BIND_CU(cuMemFree,"cuMemFree_v2");
    BIND_CU(cuImportExternalSemaphore,"cuImportExternalSemaphore");BIND_CU(cuDestroyExternalSemaphore,"cuDestroyExternalSemaphore");
    BIND_CU(cuWaitExternalSemaphoresAsync,"cuWaitExternalSemaphoresAsync");BIND_CU(cuSignalExternalSemaphoresAsync,"cuSignalExternalSemaphoresAsync");
    BIND_CU(cuGetErrorName,"cuGetErrorName");BIND_CU(cuGetErrorString,"cuGetErrorString");
#undef BIND_CU
    if(!p.cu(p.cuInitFn(0),"cuInit"))return false;
    int count=0;if(!p.cu(p.cuDeviceGetCountFn(&count),"cuDeviceGetCount"))return false;
    bool matched=false;
    for(int i=0;i<count;++i){CUdevice device=0;uint64_t luid=0;unsigned node=0;
        if(!p.cu(p.cuDeviceGetFn(&device,i),"cuDeviceGet")||!p.cu(p.cuDeviceGetLuidFn(reinterpret_cast<char*>(&luid),&node,device),"cuDeviceGetLuid"))return false;
        if(luid==context.adapter().luid&&node){p.device=device;matched=true;break;}
    }
    if(!matched){log::error("vfx-sr","CUDA/D3D12 adapter LUID mismatch");return false;}
    int major=0,minor=0;
    if(!p.cu(p.cuDeviceGetAttributeFn(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,p.device),"compute major")||
       !p.cu(p.cuDeviceGetAttributeFn(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,p.device),"compute minor"))return false;
    // Tensor cores (Turing) for every mode; the SDK lists streaming modes for Ampere or newer.
    if(major<7||(major==7&&minor<5)||(engine::vfxVideoSrNeedsAmpere(quality)&&major<8)){
        log::error("vfx-sr",std::format("mode {} not supported on compute {}.{}",quality,major,minor));return false;
    }
    if(!p.cu(p.cuDevicePrimaryCtxRetainFn(&p.cudaContext,p.device),"cuDevicePrimaryCtxRetain"))return false;p.retained=true;
    Impl::Current current(p);if(!current.active)return false;
    // A blocking stream, unlike VFG's: VideoSuperRes queues part of its work on
    // the legacy default stream, which orders only against blocking streams. On a
    // non-blocking stream it read the input before the producer fence and finished
    // after the completion signal (2026-11-05: black first frames, outputs 1-4
    // frames late with repeats and skips; aligned on every frame once blocking).
    if(!p.cu(p.cuStreamCreateFn(&p.stream,CU_STREAM_DEFAULT),"cuStreamCreate"))return false;
    const fs::path runtime=VfgBackend::runtimeDirectory(root);
    HMODULE core=nullptr,image=nullptr;
    for(auto name:{L"cudart64_12.dll",L"NVCVImage.dll",L"nvngxruntime.dll",L"NVVideoEffects.dll",L"nvVFXVideoSuperRes.dll"}){
        auto dll=load(runtime/name);if(!dll)return false;p.libraries.push_back(dll);
        if(std::wstring_view(name)==L"NVVideoEffects.dll")core=dll;if(std::wstring_view(name)==L"NVCVImage.dll")image=dll;
    }
#define BIND_VFX(member,name) if(!bind(core,name,p.member))return false
    BIND_VFX(create,"NvVFX_CreateEffect");BIND_VFX(destroy,"NvVFX_DestroyEffect");BIND_VFX(setU32,"NvVFX_SetU32");BIND_VFX(setImage,"NvVFX_SetImage");
    BIND_VFX(setStream,"NvVFX_SetCudaStream");BIND_VFX(loadEffect,"NvVFX_Load");BIND_VFX(runEffect,"NvVFX_Run");
#undef BIND_VFX
#define BIND_IMAGE(member,name) if(!bind(image,name,p.member))return false
    BIND_IMAGE(imageCreate,"NvCVImage_Create");BIND_IMAGE(imageDealloc,"NvCVImage_Dealloc");BIND_IMAGE(imageDestroy,"NvCVImage_Destroy");
    BIND_IMAGE(imageInit,"NvCVImage_Init");BIND_IMAGE(fromD3D,"NvCVImage_FromD3DFormat");BIND_IMAGE(errorString,"NvCV_GetErrorStringFromCode");
#undef BIND_IMAGE
    int pixelFormat=0,componentType=0;unsigned char layout=0;
    if(!p.cv(p.fromD3D(DXGI_FORMAT_R8G8B8A8_UNORM,&pixelFormat,&componentType,&layout),"NvCVImage_FromD3DFormat"))return false;
    if(!p.makeBuffer(p.input,inputWidth,inputHeight,pixelFormat,componentType,layout)||
       !p.makeBuffer(p.output,outputWidth,outputHeight,pixelFormat,componentType,layout))return false;
    if(!hr(context.device()->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&p.producer)),"CreateFence producer")||
       !hr(context.device()->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&p.completion)),"CreateFence completion")||
       !p.importFence(p.producer.Get(),p.inputSemaphore)||!p.importFence(p.completion.Get(),p.outputSemaphore))return false;
    const auto start=std::chrono::steady_clock::now();
    if(!p.cv(p.create("VideoSuperRes",&p.effect),"NvVFX_CreateEffect(VideoSuperRes)")||
       !p.cv(p.setImage(p.effect,"SrcImage0",p.input.image),"SrcImage0")||!p.cv(p.setImage(p.effect,"DstImage0",p.output.image),"DstImage0")||
       !p.cv(p.setStream(p.effect,"CudaStream",p.stream),"CudaStream")||!p.cv(p.setU32(p.effect,"QualityLevel",quality),"QualityLevel")||
       !p.cv(p.loadEffect(p.effect),"NvVFX_Load(VideoSuperRes)"))return false;
    p.asynchronous=GetEnvironmentVariableW(L"VEYRA_TEST_VFX_SR_SYNC",nullptr,0)==0;
    p.ready=true;
    if(p.asynchronous)p.worker=std::thread([&p]{p.work();});
    log::info("vfx-sr",std::format("VideoSuperRes quality={} input={}x{} output={}x{} async={} compute={}.{} LUID={} loadMs={:.0f}",
        quality,inputWidth,inputHeight,outputWidth,outputHeight,p.asynchronous?1:0,major,minor,context.adapter().luidString,
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()));
    return true;
#else
    (void)context;(void)root;(void)inputWidth;(void)inputHeight;(void)outputWidth;(void)outputHeight;(void)quality;
    log::error("vfx-sr","built without CUDA driver headers");return false;
#endif
}

bool VfxSrBackend::created()const{
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    return impl_->ready&&!impl_->failed.load();
#else
    return false;
#endif
}

bool VfxSrBackend::copyIn(ID3D12GraphicsCommandList* list,ID3D12Resource* texture){
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    return impl_->copy(list,texture,impl_->input,true);
#else
    (void)list;(void)texture;return false;
#endif
}

bool VfxSrBackend::copyOut(ID3D12GraphicsCommandList* list,ID3D12Resource* texture){
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    return impl_->copy(list,texture,impl_->output,false);
#else
    (void)list;(void)texture;return false;
#endif
}

bool VfxSrBackend::run(ID3D12CommandQueue* queue){
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    auto& p=*impl_;
    if(!p.ready||p.failed.load()||!queue)return false;
    if(!p.asynchronous){
        const uint64_t value=++p.signalValue;
        if(!hr(queue->Signal(p.producer.Get(),value),"signal VideoSuperRes producer"))return false;
        const bool ok=p.cudaJob(value);
        if(!ok){
            p.failed.store(true);
            std::lock_guard submit(p.cudaSubmissionMutex);
            Impl::Current current(p);
            if(current.active&&p.stream)p.cu(p.cuStreamSynchronizeFn(p.stream),"rejected VideoSuperRes drain");
            if(p.completion->GetCompletedValue()<value)
                hr(p.completion->Signal(value),"release failed VideoSuperRes job");
        }
        {
            std::lock_guard lock(p.jobsMutex);
            if(!hr(queue->Wait(p.completion.Get(),value),"queue wait VideoSuperRes completion"))return false;
            p.queuedFence=value;
            if(p.failed.load()&&p.completion->GetCompletedValue()<value)
                hr(p.completion->Signal(value),"release VideoSuperRes wait after CUDA fault");
        }
        if(!ok)p.ready=false;
        return ok;
    }
    std::lock_guard lock(p.jobsMutex);
    if(p.stopping||p.failed.load()||p.outstanding>=Impl::kMaxPending){
        log::error("vfx-sr","bounded VideoSuperRes submission rejected; no GPU wait queued");
        p.ready=false;
        return false;
    }
    const uint64_t value=++p.signalValue;
    if(!hr(queue->Signal(p.producer.Get(),value),"signal VideoSuperRes producer"))return false;
    p.jobs.push_back(value);
    if(!hr(queue->Wait(p.completion.Get(),value),"queue wait VideoSuperRes completion")){
        p.jobs.pop_back();
        return false;
    }
    p.queuedFence=value;
    ++p.outstanding;
    p.jobsChanged.notify_one();
    return true;
#else
    (void)queue;return false;
#endif
}

void VfxSrBackend::shutdown(){
#ifdef VEYRA_HAVE_CUDA_DRIVER_HEADERS
    auto& p=*impl_;p.ready=false;p.stopWorker();
    if(p.cudaContext&&p.cuCtxPushCurrentFn&&p.cuCtxPopCurrentFn){
        Impl::Current current(p);
        if(current.active){
            // Teardown, not per-frame synchronization: the graph drains its queue first.
            if(p.stream)p.cu(p.cuStreamSynchronizeFn(p.stream),"teardown cuStreamSynchronize");
            if(p.effect){p.destroy(p.effect);p.effect=nullptr;}
            p.releaseBuffer(p.output);p.releaseBuffer(p.input);
            if(p.outputSemaphore)p.cu(p.cuDestroyExternalSemaphoreFn(p.outputSemaphore),"destroy output semaphore");
            if(p.inputSemaphore)p.cu(p.cuDestroyExternalSemaphoreFn(p.inputSemaphore),"destroy input semaphore");
            if(p.stream)p.cu(p.cuStreamDestroyFn(p.stream),"cuStreamDestroy");
        }
    }
    if(p.retained)p.cu(p.cuDevicePrimaryCtxReleaseFn(p.device),"cuDevicePrimaryCtxRelease");
    p.producer.Reset();p.completion.Reset();
    for(auto i=p.libraries.rbegin();i!=p.libraries.rend();++i)FreeLibrary(*i);
    impl_=std::make_unique<Impl>();
#endif
}
}
