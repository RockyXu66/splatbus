// Example low level rendering Unity plugin

#include "PlatformBase.h"
#include "RenderAPI.h"

#include <assert.h>
#include <math.h>
#include <vector>

#include <thread>

// TCP socket headers
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif
#include <errno.h>

// JSON library (nlohmann json)
#include "nlohmann/json.hpp"

// Debug log callback
using DebugLogFunc = void (*)(const char*);
DebugLogFunc g_DebugLogCallback = nullptr;
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetDebugLogCallback(void* callback)
{
    g_DebugLogCallback = (DebugLogFunc)callback;
}

// --------------------------------------------------------------------------
// SetTimeFromUnity, an example function we export which is called by one of the scripts.

static float g_Time;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTimeFromUnity (float t) { g_Time = t; }



// --------------------------------------------------------------------------
// SetTextureFromUnity, an example function we export which is called by one of the scripts.

static void* g_TextureHandle = NULL;
static int   g_TextureWidth  = 0;
static int   g_TextureHeight = 0;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int w, int h)
{
	// A script calls this at initialization time; just remember the texture pointer here.
	// Will update texture pixels each frame from the plugin rendering event (texture update
	// needs to happen on the rendering thread).
	g_TextureHandle = textureHandle;
	g_TextureWidth = w;
	g_TextureHeight = h;
}


// --------------------------------------------------------------------------
// SetMeshBuffersFromUnity, an example function we export which is called by one of the scripts.

static void* g_VertexBufferHandle = NULL;
static int g_VertexBufferVertexCount;

struct MeshVertex
{
	float pos[3];
	float normal[3];
	float color[4];
	float uv[2];
};
static std::vector<MeshVertex> g_VertexSource;


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetMeshBuffersFromUnity(void* vertexBufferHandle, int vertexCount, float* sourceVertices, float* sourceNormals, float* sourceUV)
{
	// A script calls this at initialization time; just remember the pointer here.
	// Will update buffer data each frame from the plugin rendering event (buffer update
	// needs to happen on the rendering thread).
	g_VertexBufferHandle = vertexBufferHandle;
	g_VertexBufferVertexCount = vertexCount;

	// The script also passes original source mesh data. The reason is that the vertex buffer we'll be modifying
	// will be marked as "dynamic", and on many platforms this means we can only write into it, but not read its previous
	// contents. In this example we're not creating meshes from scratch, but are just altering original mesh data --
	// so remember it. The script just passes pointers to regular C# array contents.
	g_VertexSource.resize(vertexCount);
	for (int i = 0; i < vertexCount; ++i)
	{
		MeshVertex& v = g_VertexSource[i];
		v.pos[0] = sourceVertices[0];
		v.pos[1] = sourceVertices[1];
		v.pos[2] = sourceVertices[2];
		v.normal[0] = sourceNormals[0];
		v.normal[1] = sourceNormals[1];
		v.normal[2] = sourceNormals[2];
		v.uv[0] = sourceUV[0];
		v.uv[1] = sourceUV[1];
		sourceVertices += 3;
		sourceNormals += 3;
		sourceUV += 2;
	}
}


// --------------------------------------------------------------------------
// UnitySetInterfaces

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
	s_UnityInterfaces = unityInterfaces;
	s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
	s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);
	
#if SUPPORT_VULKAN
	if (s_Graphics->GetRenderer() == kUnityGfxRendererNull)
	{
		extern void RenderAPI_Vulkan_OnPluginLoad(IUnityInterfaces*);
		RenderAPI_Vulkan_OnPluginLoad(unityInterfaces);
	}
#endif // SUPPORT_VULKAN

	// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
	OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
	s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

#if UNITY_WEBGL
typedef void	(UNITY_INTERFACE_API * PluginLoadFunc)(IUnityInterfaces* unityInterfaces);
typedef void	(UNITY_INTERFACE_API * PluginUnloadFunc)();

extern "C" void	UnityRegisterRenderingPlugin(PluginLoadFunc loadPlugin, PluginUnloadFunc unloadPlugin);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API RegisterPlugin()
{
	UnityRegisterRenderingPlugin(UnityPluginLoad, UnityPluginUnload);
}
#endif

// --------------------------------------------------------------------------
// GraphicsDeviceEvent


static RenderAPI* s_CurrentAPI = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;


static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
	// Create graphics API implementation upon initialization
	if (eventType == kUnityGfxDeviceEventInitialize)
	{
		assert(s_CurrentAPI == NULL);
		s_DeviceType = s_Graphics->GetRenderer();
		s_CurrentAPI = CreateRenderAPI(s_DeviceType);
	}

	// Let the implementation process the device related events
	if (s_CurrentAPI)
	{
		s_CurrentAPI->ProcessDeviceEvent(eventType, s_UnityInterfaces);
	}

	// Cleanup graphics API implementation upon shutdown
	if (eventType == kUnityGfxDeviceEventShutdown)
	{
		delete s_CurrentAPI;
		s_CurrentAPI = NULL;
		s_DeviceType = kUnityGfxRendererNull;
	}
}



// --------------------------------------------------------------------------
// OnRenderEvent
// This will be called for GL.IssuePluginEvent script calls; eventID will
// be the integer passed to IssuePluginEvent. In this example, we just ignore
// that value.


static void DrawColoredTriangle()
{
	// Draw a colored triangle. Note that colors will come out differently
	// in D3D and OpenGL, for example, since they expect color bytes
	// in different ordering.
	struct MyVertex
	{
		float x, y, z;
		unsigned int color;
	};
	MyVertex verts[3] =
	{
		{ -0.5f, -0.25f,  0, 0xFFff0000 },
		{ 0.5f, -0.25f,  0, 0xFF00ff00 },
		{ 0,     0.5f ,  0, 0xFF0000ff },
	};

	// Transformation matrix: rotate around Z axis based on time.
	float phi = g_Time; // time set externally from Unity script
	float cosPhi = cosf(phi);
	float sinPhi = sinf(phi);
	float depth = 0.7f;
	float finalDepth = s_CurrentAPI->GetUsesReverseZ() ? 1.0f - depth : depth;
	float worldMatrix[16] = {
		cosPhi,-sinPhi,0,0,
		sinPhi,cosPhi,0,0,
		0,0,1,0,
		0,0,finalDepth,1,
	};

	s_CurrentAPI->DrawSimpleTriangles(worldMatrix, 1, verts);
}


static void ModifyTexturePixels()
{
	void* textureHandle = g_TextureHandle;
	int width = g_TextureWidth;
	int height = g_TextureHeight;
	if (!textureHandle)
		return;

	int textureRowPitch;
	void* textureDataPtr = s_CurrentAPI->BeginModifyTexture(textureHandle, width, height, &textureRowPitch);
	if (!textureDataPtr)
		return;

	const float t = g_Time * 4.0f;

	unsigned char* dst = (unsigned char*)textureDataPtr;
	for (int y = 0; y < height; ++y)
	{
		unsigned char* ptr = dst;
		for (int x = 0; x < width; ++x)
		{
			// Simple "plasma effect": several combined sine waves
			int vv = int(
				(127.0f + (127.0f * sinf(x / 7.0f + t))) +
				(127.0f + (127.0f * sinf(y / 5.0f - t))) +
				(127.0f + (127.0f * sinf((x + y) / 6.0f - t))) +
				(127.0f + (127.0f * sinf(sqrtf(float(x*x + y*y)) / 4.0f - t)))
				) / 4;

			// Write the texture pixel
			ptr[0] = vv;
			ptr[1] = vv;
			ptr[2] = vv;
			ptr[3] = vv;

			// To next pixel (our pixels are 4 bpp)
			ptr += 4;
		}

		// To next image row
		dst += textureRowPitch;
	}

	s_CurrentAPI->EndModifyTexture(textureHandle, width, height, textureRowPitch, textureDataPtr);
}


static void ModifyVertexBuffer()
{
	void* bufferHandle = g_VertexBufferHandle;
	int vertexCount = g_VertexBufferVertexCount;
	if (!bufferHandle)
		return;

	size_t bufferSize;
	void* bufferDataPtr = s_CurrentAPI->BeginModifyVertexBuffer(bufferHandle, &bufferSize);
	if (!bufferDataPtr)
		return;
	int vertexStride = int(bufferSize / vertexCount);

	// Unity should return us a buffer that is the size of `vertexCount * sizeof(MeshVertex)`
	// If that's not the case then we should quit to avoid unexpected results.
	// This can happen if https://docs.unity3d.com/ScriptReference/Mesh.GetNativeVertexBufferPtr.html returns
	// a pointer to a buffer with an unexpected layout.
	if (static_cast<unsigned int>(vertexStride) != sizeof(MeshVertex))
		return;

	const float t = g_Time * 3.0f;

	char* bufferPtr = (char*)bufferDataPtr;
	// modify vertex Y position with several scrolling sine waves,
	// copy the rest of the source data unmodified
	for (int i = 0; i < vertexCount; ++i)
	{
		const MeshVertex& src = g_VertexSource[i];
		MeshVertex& dst = *(MeshVertex*)bufferPtr;
		dst.pos[0] = src.pos[0];
		dst.pos[1] = src.pos[1] + sinf(src.pos[0] * 1.1f + t) * 0.4f + sinf(src.pos[2] * 0.9f - t) * 0.3f;
		dst.pos[2] = src.pos[2];
		dst.normal[0] = src.normal[0];
		dst.normal[1] = src.normal[1];
		dst.normal[2] = src.normal[2];
		dst.uv[0] = src.uv[0];
		dst.uv[1] = src.uv[1];
		bufferPtr += vertexStride;
	}

	s_CurrentAPI->EndModifyVertexBuffer(bufferHandle);
}

static void drawToPluginTexture()
{
	s_CurrentAPI->drawToPluginTexture();
}

static void drawToRenderTexture()
{
	s_CurrentAPI->drawToRenderTexture();
}


// --------------------------------------------------------------------------
// DX12 plugin specific
// --------------------------------------------------------------------------

extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetRenderTexture()
{
	return s_CurrentAPI->getRenderTexture();
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetRenderTexture(UnityRenderBuffer rb)
{
	s_CurrentAPI->setRenderTextureResource(rb);
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsSwapChainAvailable()
{
	return s_CurrentAPI->isSwapChainAvailable();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetPresentFlags()
{
	return s_CurrentAPI->getPresentFlags();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetSyncInterval()
{
	return s_CurrentAPI->getSyncInterval();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetBackBufferWidth()
{
	return s_CurrentAPI->getBackbufferHeight();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetBackBufferHeight()
{
	return s_CurrentAPI->getBackbufferWidth();
}

















// --------------------------------------------------------------------------
// IPC functionality for Gaussian Splatting
// --------------------------------------------------------------------------

// For IPC functionality, we need OpenGL and CUDA interop
#if SUPPORT_OPENGL_UNIFIED
    #include "gl3w/gl3w.h"
    #include <cuda_gl_interop.h>
#endif

using json = nlohmann::json;

static std::atomic<bool> gIsConnectedToServer{false};
static std::thread gServerThread;

static cudaGraphicsResource* gColorRes = nullptr;
static void*  gIpcColorPtr = nullptr;
static size_t gPitchColor  = 0;
static cudaGraphicsResource* gDepthRes = nullptr;
static void*  gIpcDepthPtr = nullptr;
static size_t gPitchDepth  = 0;
static int    gW=0, gH=0;
static cudaEvent_t gFrameDone = nullptr;
static cudaStream_t gStream = 0;
static void* gIpcBasePtr = nullptr; // Base pointer for shared IPC allocation

// For SetUnityTargets parameters
static void* gPendingTexture = nullptr;
static void* gPendingDepthTexture = nullptr;
static int gPendingW = 0, gPendingH = 0;
static bool gNeedsTargetSetup = false;

// --- Simple base64 decoding ---
static std::vector<unsigned char> b64decode(const std::string& s);

// --- socket helpers ---
#ifdef _WIN32
using socket_t = SOCKET;
static const socket_t kInvalidSocket = INVALID_SOCKET;
static bool socket_init() {
    WSADATA wsaData;
    return WSAStartup(MAKEWORD(2, 2), &wsaData) == 0;
}
static void socket_cleanup() { WSACleanup(); }
static void socket_close(socket_t s) { closesocket(s); }
static int socket_error_code() { return WSAGetLastError(); }
#else
using socket_t = int;
static const socket_t kInvalidSocket = -1;
static bool socket_init() { return true; }
static void socket_cleanup() {}
static void socket_close(socket_t s) { ::close(s); }
static int socket_error_code() { return errno; }
#endif

// --- Length prefix reading ---
static bool recv_packet(socket_t fd, std::vector<char>& out) {
    uint32_t n=0;
    if (::recv(fd, reinterpret_cast<char*>(&n), 4, MSG_WAITALL) != 4) return false;
    out.resize(n);
    return (::recv(fd, out.data(), n, MSG_WAITALL) == (int)n);
}

// --- A simple base64 decoding implementation (RFC4648, no line breaks) ---
static inline int b64val(unsigned char c){
    if(c>='A'&&c<='Z') return c-'A';
    if(c>='a'&&c<='z') return c-'a'+26;
    if(c>='0'&&c<='9') return c-'0'+52;
    if(c=='+') return 62; if(c=='/') return 63; if(c=='=') return -1;
    return -2;
}
static std::vector<unsigned char> b64decode(const std::string& s){
    std::vector<unsigned char> out; out.reserve(s.size()*3/4);
    int val=0, valb=-8;
    for(unsigned char c: s){
        int d = b64val(c); if(d==-2) continue; // skip whitespace
        if(d==-1){ /* padding */ break; }
        val = (val<<6) + d; valb += 6;
        if(valb>=0){ out.push_back((unsigned char)((val>>valb)&0xFF)); valb-=8; }
    }
    return out;
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GsIpc_ConnectToPythonServer() 
{
    if (gIsConnectedToServer.exchange(true)) return; // Already connected to server

    gServerThread = std::thread([]{
        // As client, connect to Python server 127.0.0.1:6001, receive INIT packet
        const char* SERVER_HOST = "127.0.0.1";
        const uint16_t SERVER_PORT = 6001;

        if (!socket_init()) { gIsConnectedToServer=false; return; }

        std::stringstream ss;
        ss.str("");
        ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> Connecting to " << SERVER_HOST << ":" << SERVER_PORT << std::endl;
        LogMessage(ss);

        // Main client loop - connect and receive init, reconnect if needed
        while (gIsConnectedToServer) {
            socket_t cli = ::socket(AF_INET, SOCK_STREAM, 0);
            if (cli == kInvalidSocket) {
                ss.str("");
                ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> socket create failed: " << socket_error_code() << std::endl;
                LogMessage(ss);
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(SERVER_PORT);
            addr.sin_addr.s_addr = inet_addr(SERVER_HOST);

            if (::connect(cli, (sockaddr*)&addr, sizeof(addr)) < 0) {
                ss.str("");
                ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> connect failed: " << socket_error_code() << std::endl;
                LogMessage(ss);
                socket_close(cli);
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            ss.str("");
            ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> Connected to Python server" << std::endl;
            LogMessage(ss);

            // Receive INIT packet (JSON)
            std::vector<char> blob;
            if (!recv_packet(cli, blob)) { 
                socket_close(cli); 
                ss.str("");
                ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> Failed to receive INIT packet, waiting for next client..." << std::endl;
                LogMessage(ss);
                continue; 
            }

            json j = json::parse(blob.begin(), blob.end(), nullptr, /*allow_exceptions=*/true);

            // Parse meta data
            auto meta = j.at("meta");
            gW = meta.at("w").get<int>();
            gH = meta.at("h").get<int>();

            if (meta.contains("device")) {
                int device = meta.at("device").get<int>();
                cudaError_t devErr = cudaSetDevice(device);
                if (devErr != cudaSuccess) {
                    ss.str("");
                    ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error -> cudaSetDevice(" << device
                       << "): " << cudaGetErrorString(devErr) << std::endl;
                    LogMessage(ss);
                    socket_close(cli);
                    continue;
                }
                // Ensure CUDA context is initialized on this thread/device
                cudaError_t initErr = cudaFree(0);
                if (initErr != cudaSuccess) {
                    ss.str("");
                    ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error -> cudaFree(0): "
                       << cudaGetErrorString(initErr) << std::endl;
                    LogMessage(ss);
                    socket_close(cli);
                    continue;
                }
            }
            
            ss.str("");
            ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> gW: " << gW << "; gH: " << gH 
               << "; pitchColor: " << gPitchColor << "; pitchDepth: " << gPitchDepth << std::endl;
            LogMessage(ss);

            std::string fmtColor = meta.at("fmtColor").get<std::string>(); // Expected "RGBA32F"
            gPitchColor = meta.at("pitchColor").get<size_t>();
            
            std::string fmtDepth = meta.at("fmtDepth").get<std::string>(); // Expected "R32F"
            gPitchDepth = meta.at("pitchDepth").get<size_t>();
            uint64_t ptrColor = 0;
            uint64_t ptrDepth = 0;
            if (meta.contains("ptrColor")) {
                ptrColor = meta.at("ptrColor").get<uint64_t>();
            }
            if (meta.contains("ptrDepth")) {
                ptrDepth = meta.at("ptrDepth").get<uint64_t>();
            }

            // Parse base64 handles
            std::vector<unsigned char> mem_color = b64decode(j.at("mem_color").get<std::string>());
            std::vector<unsigned char> mem_depth = b64decode(j.at("mem_depth").get<std::string>());
            std::vector<unsigned char> evt_done  = b64decode(j.at("evt_done").get<std::string>());
            if (mem_color.size() != sizeof(cudaIpcMemHandle_t) || mem_depth.size() != sizeof(cudaIpcMemHandle_t) || evt_done.size() != sizeof(cudaIpcEventHandle_t)) {
                // fprintf(stderr, "IPC handle size mismatch\n");
                ss.str("");
                ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error -> IPC handle size mismatch" << std::endl;
                LogMessage(ss);
                socket_close(cli);
                continue; // Wait for next client instead of shutting down
            }

            // Close previous IPC handles if they exist
            if (gIpcBasePtr) {
                cudaIpcCloseMemHandle(gIpcBasePtr);
                gIpcBasePtr = nullptr;
            } else {
                if (gIpcColorPtr) {
                    cudaIpcCloseMemHandle(gIpcColorPtr);
                }
                if (gIpcDepthPtr) {
                    cudaIpcCloseMemHandle(gIpcDepthPtr);
                }
            }
            gIpcColorPtr = nullptr;
            gIpcDepthPtr = nullptr;

            // Open IPC handles
            cudaIpcMemHandle_t mhc; std::memcpy(&mhc, mem_color.data(), sizeof(mhc));
            cudaIpcMemHandle_t mhd; std::memcpy(&mhd, mem_depth.data(), sizeof(mhd));
            cudaIpcEventHandle_t eh; std::memcpy(&eh,  evt_done.data(),  sizeof(eh));

            // Clear any sticky error before opening IPC handles
            cudaGetLastError();
            cudaError_t e2 = cudaIpcOpenEventHandle(&gFrameDone, eh);
            if (e2 != cudaSuccess) 
            { 
                ss.str("");
                ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error  -> cudaIpcOpenEventHandle: " << cudaGetErrorString(e2) << std::endl;
                LogMessage(ss);
                socket_close(cli);
                continue; // Wait for next client instead of shutting down
            }

            const bool sharedHandle = (mem_color == mem_depth);
            if (sharedHandle) {
                if (ptrColor == 0 || ptrDepth == 0) {
                    ss.str("");
                    ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error -> Shared IPC handle but missing ptrColor/ptrDepth" << std::endl;
                    LogMessage(ss);
                    socket_close(cli);
                    continue;
                }
                cudaError_t e1 = cudaIpcOpenMemHandle(&gIpcBasePtr, mhc, cudaIpcMemLazyEnablePeerAccess);
                if (e1 != cudaSuccess) 
                { 
                    ss.str("");
                    ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error -> cudaIpcOpenMemHandle(shared): " << cudaGetErrorString(e1) << std::endl;
                    LogMessage(ss);
                    socket_close(cli);
                    continue; // Wait for next client instead of shutting down
                }
                const intptr_t depthOffset = (intptr_t)ptrDepth - (intptr_t)ptrColor;
                gIpcColorPtr = gIpcBasePtr;
                gIpcDepthPtr = (void*)((uintptr_t)gIpcBasePtr + (uintptr_t)depthOffset);
                ss.str("");
                ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> Shared IPC handle, depth offset: " << depthOffset << " bytes" << std::endl;
                LogMessage(ss);
            } else {
                cudaError_t e1 = cudaIpcOpenMemHandle(&gIpcColorPtr, mhc, cudaIpcMemLazyEnablePeerAccess);
                if (e1 != cudaSuccess) 
                { 
                    ss.str("");
                    ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error -> cudaIpcOpenMemHandle(color): " << cudaGetErrorString(e1) << std::endl;
                    LogMessage(ss);
                    socket_close(cli);
                    continue; // Wait for next client instead of shutting down
                }
                cudaError_t e3 = cudaIpcOpenMemHandle(&gIpcDepthPtr, mhd, cudaIpcMemLazyEnablePeerAccess);
                if (e3 != cudaSuccess) 
                { 
                    ss.str("");
                    ss << "RenderPlugin::GsIpc_ConnectToPythonServer() Error -> cudaIpcOpenMemHandle(depth): " << cudaGetErrorString(e3) << std::endl;
                    LogMessage(ss);
                    socket_close(cli);
                    continue; // Wait for next client instead of shutting down
                }
            }

            // Optional: subsequent frames META (not enforced, can be read and discarded)
            std::vector<char> frameMeta;
            while (gIsConnectedToServer) {
                if (!recv_packet(cli, frameMeta)) break;
                // json fm = json::parse(frameMeta.begin(), frameMeta.end());
                // Not used, can be ignored
            }
            socket_close(cli);
            
            ss.str("");
            ss << "RenderPlugin::GsIpc_ConnectToPythonServer() -> Disconnected from Python server, retrying..." << std::endl;
            LogMessage(ss);
        }

        // Clean up socket system when shutting down
        socket_cleanup();
        gIsConnectedToServer=false;
    });
    gServerThread.detach();
}

// Called from main thread, set pending texture parameters
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GsIpc_PrepareUnityTargets(void* glColorTexName, void* glDepthTexName, int w, int h)
{
    gPendingTexture = glColorTexName;
    gPendingDepthTexture = glDepthTexName;
    gPendingW = w;
    gPendingH = h;
    gNeedsTargetSetup = true;
}

// Function executed in rendering thread
static void GsIpc_SetUnityTargets(void* glColorTexName, void* glDepthTexName, int w, int h)
{
    gW=w; gH=h;
    GLuint tex = (GLuint)(uintptr_t)glColorTexName;
	GLuint depthTex = (GLuint)(uintptr_t)glDepthTexName;
    
    std::stringstream ss;
    ss << "RenderPlugin::GsIpc_SetUnityTargets() -> tex: " << tex << "; w: " << w << "; h: " << h << std::endl;
    LogMessage(ss);
    
    // Register OpenGL color texture (RGBA32F)
    cudaError_t err = cudaGraphicsGLRegisterImage(&gColorRes, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        ss.str("");
        ss << "RenderPlugin::GsIpc_SetUnityTargets() -> cudaGraphicsGLRegisterImage failed: " << cudaGetErrorString(err) << std::endl;
        LogMessage(ss);
        return;
    }
    // Register OpenGL depth texture (R32F)
    err = cudaGraphicsGLRegisterImage(&gDepthRes, depthTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        ss.str("");
        ss << "RenderPlugin::GsIpc_SetUnityTargets() -> cudaGraphicsGLRegisterImage failed: " << cudaGetErrorString(err) << std::endl;
        LogMessage(ss);
        return;
    }
    cudaStreamCreate(&gStream);
    
    ss.str("");
    ss << "RenderPlugin::GsIpc_SetUnityTargets() -> Successfully registered texture" << std::endl;
    LogMessage(ss);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GsIpc_UpdateFrame() 
{
    std::stringstream ss;
    // ss.str("");
    // ss << "Call ----> RenderPlugin::GsIpc_UpdateFrame() " << std::endl;
    // LogMessage(ss);
	// if (!gFrameDone)
	// {
	// 	ss.str("");
	// 	ss << "RenderPlugin::GsIpc_UpdateFrame() [No gFrameDone]" << std::endl;
	// 	LogMessage(ss);
	// }
	// if (!gColorRes)
	// {
	// 	ss.str("");
	// 	ss << "RenderPlugin::GsIpc_UpdateFrame() [No gColorRes]" << std::endl;
	// 	LogMessage(ss);
	// }
	// if (!gIpcColorPtr)
	// {
	// 	ss.str("");
	// 	ss << "RenderPlugin::GsIpc_UpdateFrame() [No gIpcColorPtr]" << std::endl;
	// 	LogMessage(ss);
	// }

    if (!gFrameDone || !gColorRes || !gIpcColorPtr || !gDepthRes || !gIpcDepthPtr) return;

    // ss.str("");
    // ss << "RenderPlugin::GsIpc_UpdateFrame() " << std::endl;
    // LogMessage(ss);
    cudaStreamWaitEvent(gStream, gFrameDone, 0);

    cudaGraphicsMapResources(1, &gColorRes, gStream);
    cudaGraphicsMapResources(1, &gDepthRes, gStream);
    cudaArray_t arrC=nullptr;
    cudaArray_t arrD=nullptr;
    cudaGraphicsSubResourceGetMappedArray(&arrC, gColorRes, 0, 0);
    cudaGraphicsSubResourceGetMappedArray(&arrD, gDepthRes, 0, 0);

    size_t widthBytes = (size_t)gW * 4 * sizeof(float); // RGBA32F
    cudaMemcpy2DToArrayAsync(arrC, 0, 0,
                             gIpcColorPtr, gPitchColor,
                             widthBytes, (size_t)gH,
                             cudaMemcpyDeviceToDevice, gStream);
    size_t depthWidthBytes = (size_t)gW * sizeof(float); // R32F
    cudaMemcpy2DToArrayAsync(arrD, 0, 0,
                             gIpcDepthPtr, gPitchDepth,
                             depthWidthBytes, (size_t)gH,
                             cudaMemcpyDeviceToDevice, gStream);

    cudaGraphicsUnmapResources(1, &gColorRes, gStream);
    cudaGraphicsUnmapResources(1, &gDepthRes, gStream);
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	// Unknown / unsupported graphics device type? Do nothing
	if (s_CurrentAPI == NULL)
		return;

	if (eventID == 1)
	{
        drawToRenderTexture();
        DrawColoredTriangle();
        ModifyTexturePixels();
        ModifyVertexBuffer();
	}

	if (eventID == 2)
	{
		drawToPluginTexture();
	}
	
	if (eventID == RenderAPI::IPC_SETUP_TARGETS)
	{
		if (gNeedsTargetSetup && gPendingTexture != nullptr && gPendingDepthTexture != nullptr)
		{
			GsIpc_SetUnityTargets(gPendingTexture, gPendingDepthTexture, gPendingW, gPendingH);
			gNeedsTargetSetup = false;
		}
	}
	if (eventID == RenderAPI::IPC_UPDATE_FRAME)
	{
		GsIpc_UpdateFrame();
	}


}

// --------------------------------------------------------------------------
// GetRenderEventFunc, an example function we export which is used to get a rendering event callback function.

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
	return OnRenderEvent;
}
