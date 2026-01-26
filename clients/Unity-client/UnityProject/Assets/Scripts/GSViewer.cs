using UnityEngine;
using System;
using System.Collections;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;


public class GSViewer : MonoBehaviour
{
    private Camera cam;
    private Coroutine initCo = null;

    [Header("Dynamic Parameters")]
    [Range(0.1f, 1f)]
    public float texFactor = 1.0f;

    public bool isXr = false;       // Not supported yet
    public Material renderMaterial;
    // public Material depthMaterial;
    
    [Header("IPC Settings")]
    public bool useIpc = true;
    public int ipcRenderWidth = 532;
    public int ipcRenderHeight = 948;

    [Header("Debug Settings")]
    public bool showDebugGUI = false;
    
    private RenderTexture ipcRenderTexture;
    private RenderTexture ipcDepthTexture;
    private bool ipcTargetSetup = false;

#if (PLATFORM_IOS || PLATFORM_TVOS || PLATFORM_BRATWURST || PLATFORM_SWITCH) && !UNITY_EDITOR
    [DllImport("__Internal")]
#else
    [DllImport("RenderingPlugin")]
#endif
    private static extern IntPtr GetRenderEventFunc();

    public class Native
    {
        public const int IPC_UPDATE_FRAME = 0x0100;
        public const int IPC_SETUP_TARGETS = 0x0101;
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] public delegate void DebugLogDelegate(string message);
        [DllImport("RenderingPlugin", EntryPoint = "SetDebugLogCallback")] public static extern void SetDebugLogCallback(IntPtr callback);

        // IPC functions
        [DllImport("RenderingPlugin", EntryPoint = "GsIpc_ConnectToPythonServer")] 
        public static extern void GsIpc_ConnectToPythonServer();
        
        [DllImport("RenderingPlugin", EntryPoint = "GsIpc_PrepareUnityTargets")] 
        public static extern void GsIpc_PrepareUnityTargets(IntPtr texturePtr, IntPtr depthTexturePtr, int width, int height);
    }

    #region Members
    protected IntPtr renderEventFunc = IntPtr.Zero;
    #endregion

    protected IEnumerator Initialize()
    {
        Vector2Int resolution;
        resolution = new Vector2Int((int)(cam.pixelWidth * texFactor), (int)(cam.pixelHeight * texFactor));

        if (renderEventFunc == IntPtr.Zero)
        {
            renderEventFunc = GetRenderEventFunc();
        }

        if (renderEventFunc == IntPtr.Zero)
		{
			Debug.LogError("Cannot get render event function! Exiting...");
			yield break;
		}

        // === IPC initialization ===
        if (useIpc)
        {
            Debug.Log("IPC Mode: Connecting to IPC Server...");
            
            Native.GsIpc_ConnectToPythonServer();
            
            // Create IPC RenderTexture (RGBA32F format for color, RFloat format for depth)
            ipcRenderTexture = new RenderTexture(ipcRenderWidth, ipcRenderHeight, 0, RenderTextureFormat.ARGBFloat);
            ipcDepthTexture = new RenderTexture(ipcRenderWidth, ipcRenderHeight, 0, RenderTextureFormat.RFloat);
            ipcRenderTexture.Create();
            ipcDepthTexture.Create();
            if (!ipcRenderTexture.IsCreated())
            {
                Debug.LogError("Failed to create IPC RenderTexture!");
                useIpc = false;
            }
            else if (!ipcDepthTexture.IsCreated())
            {
                Debug.LogError("Failed to create IPC Depth RenderTexture!");
                useIpc = false;
            }
            else
            {
                Debug.Log($"Created IPC RenderTexture: {ipcRenderWidth}x{ipcRenderHeight}");
                
                // Prepare texture parameters (on main thread)
                IntPtr texturePtr = ipcRenderTexture.GetNativeTexturePtr();
                IntPtr depthTexturePtr = ipcDepthTexture.GetNativeTexturePtr();
                Native.GsIpc_PrepareUnityTargets(texturePtr, depthTexturePtr, ipcRenderWidth, ipcRenderHeight);
                
                Debug.Log($"Prepared IPC targets with texture ptr: {texturePtr}");
                Debug.Log($"Prepared IPC targets with depth texture ptr: {depthTexturePtr}");
            }
        }
    }

    protected void Awake()
    {			
        Debug.Log("UseRenderingPlugin::Awake()");
        // Get a function pointer for our static log method for debugging
        Native.DebugLogDelegate logDelegate = new Native.DebugLogDelegate(NativeDebugLog);
        IntPtr logCallbackPtr = Marshal.GetFunctionPointerForDelegate(logDelegate);

        // Pass the pointer to the native plugin
        Native.SetDebugLogCallback(logCallbackPtr);
    }

    private void OnEnable()
    {
        cam = GetComponent<Camera>();
        
        // Make sure the Camera is rendering depth texture (shader needs _CameraDepthTexture)
        cam.depthTextureMode = DepthTextureMode.Depth;
        
        initCo = StartCoroutine(Initialize());
    }

    void OnDisable()
    {
        // Clean up IPC resources
        if (ipcRenderTexture != null)
        {
            ipcRenderTexture.Release();
            Destroy(ipcRenderTexture);
            ipcRenderTexture = null;
            Debug.Log("IPC RenderTexture released");
        }
        if (ipcDepthTexture != null)
        {
            ipcDepthTexture.Release();
            Destroy(ipcDepthTexture);
            ipcDepthTexture = null;
            Debug.Log("IPC Depth Texture released");
        }
    }


    [ImageEffectOpaque]
	protected void OnRenderImage(RenderTexture source, RenderTexture destination)
	{
		// === IPC mode processing ===
		if (useIpc && ipcRenderTexture != null && ipcRenderTexture.IsCreated())
		{
			// Register textures (only once)
			if (!ipcTargetSetup)
			{
				SendIpcSetupTargetsEvent();
				ipcTargetSetup = true;
				Debug.Log("IPC targets setup issued");
			}
			
			// Copy data from Python CUDA memory to Unity texture every frame
			SendIpcUpdateFrameEvent();

			// Display IPC rendered content
			if (renderMaterial != null)
			{
				renderMaterial.SetFloat("_Scale", 1.0f);
				renderMaterial.SetTexture("_GaussianSplattingTexLeftEye", ipcRenderTexture);
				renderMaterial.SetTexture("_GaussianSplattingDepthTexLeftEye", ipcDepthTexture);
				Graphics.Blit(source, destination, renderMaterial);
			}
			else
			{
				// If no material, directly display IPC content
				Graphics.Blit(ipcRenderTexture, destination);
			}
			return;
		}
        else
        {
            Graphics.Blit(source, destination);
            return;
        }
	}

    public void SendIpcSetupTargetsEvent()
    {
        GL.IssuePluginEvent(renderEventFunc, Native.IPC_SETUP_TARGETS);
    }

    public void SendIpcUpdateFrameEvent()
    {
        GL.IssuePluginEvent(renderEventFunc, Native.IPC_UPDATE_FRAME);
        GL.InvalidateState();
    }

    #region Native Logging
    //  This is the static method that will be called from C++
    [AOT.MonoPInvokeCallback(typeof(Native.DebugLogDelegate))]
    private static void NativeDebugLog(string message)
    {
        Debug.Log($"[NativePlugin]: {message}");
    }
    #endregion
    
    #region IPC Debug Info
    void OnGUI()
    {
        if (!showDebugGUI) return;

        if (useIpc)
        {
            GUILayout.BeginArea(new Rect(10, 10, 350, 150));
            GUILayout.Box("=== IPC Status ===");
            GUILayout.Label($"IPC Mode: {(useIpc ? "Enabled" : "Disabled")}");
            GUILayout.Label($"Target Setup: {(ipcTargetSetup ? "✓ Ready" : "Initializing...")}");
            GUILayout.Label($"Resolution: {ipcRenderWidth}x{ipcRenderHeight}");
            if (ipcRenderTexture != null)
            {
                GUILayout.Label($"RenderTexture: {(ipcRenderTexture.IsCreated() ? "✓ Created" : "✗ Failed")}");
            }
            else
            {
                GUILayout.Label("RenderTexture: Not initialized");
            }
            if (ipcDepthTexture != null)
            {
                GUILayout.Label($"DepthTexture: {(ipcDepthTexture.IsCreated() ? "✓ Created" : "✗ Failed")}");
            }
            else
            {
                GUILayout.Label("DepthTexture: Not initialized");
            }
            GUILayout.EndArea();
        }
    }
    #endregion
}