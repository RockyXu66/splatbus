using System;
using UnityEngine;

[DisallowMultipleComponent]
public class UnityDataSender : MonoBehaviour
{
    public GameObject pointCloud;

    [Header("Coordinate Conversion")]
    [Tooltip("Convert Unity coordinates to OpenCV coordinates (flip Y axis) before sending")]
    public bool convertToOpenCV = true;

    [Header("Connection")]
    public string host = "127.0.0.1";
    public int port = 6000;
    public bool autoReconnect = true;

    [Header("Send Settings")]
    [Tooltip("Camera to sample. Defaults to the Camera on this GameObject.")]
    public Camera targetCamera;
    [Tooltip("Seconds between pose updates.")]
    [Range(0.01f, 0.5f)]
    public float sendInterval = 0.02f;

    private float elapsed;
    private SplatbusMessageClient client;

    private const string LogTag = "[UnityDataSender]";

    private void Awake()
    {
        client = new SplatbusMessageClient();
    }

    private void Start()
    {
        if (targetCamera == null)
        {
            targetCamera = GetComponent<Camera>();
        }
        if (pointCloud == null)
        {
            pointCloud = GameObject.Find("PointClouds");
        }

        TryConnect();

        if (client.IsConnected && targetCamera != null)
        {
            int cam_idx = 0;
            GetCameraPose(targetCamera, cam_idx);
            int[] size = GetViewportSize();
            Debug.Log($"{LogTag} Viewpoint size: {size[0]}x{size[1]}");

            int width = size[0];
            int height = size[1];
            if (width != targetCamera.pixelWidth || height != targetCamera.pixelHeight)
            {
                Debug.LogError(
                    $"{LogTag} Viewport size mismatch. [Renderer: {width}x{height}] " +
                    $" [Unity: {targetCamera.pixelWidth}x{targetCamera.pixelHeight}]" +
                    $" -> Please adjust the viewport size in the renderer or the viewport size in Unity."
                );
                QuitWithError();
                return;
            }
        }
    }

    private void QuitWithError()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit(1);
#endif
    }

    private void Update()
    {
        if (!client.IsConnected)
        {
            if (autoReconnect)
            {
                TryConnect();
            }
            return;
        }

        elapsed += Time.unscaledDeltaTime;
        if (elapsed >= sendInterval)
        {
            elapsed = 0f;
            SendCameraPose();
            SendPointCloudPose();
        }
    }

    private void TryConnect()
    {
        client.Close();

        try
        {
            client.Connect(host, port);
            Debug.Log($"{LogTag} Connected to {host}:{port}");
        }
        catch (Exception)
        {
            // Debug.LogWarning($"{LogTag} Failed to connect to {host}:{port} ({ex.Message})");
            client.Close();
        }
    }

    private void SendCameraPose()
    {
        if (!client.IsConnected)
        {
            return;
        }

        float timestamp = Time.realtimeSinceStartup;

        if (targetCamera != null)
        {
            SendTransform(
                targetCamera.transform,
                "camera_pose",
                timestamp
            );
        }
    }

    private void SendPointCloudPose()
    {
        if (!client.IsConnected)
        {
            return;
        }

        float timestamp = Time.realtimeSinceStartup;

        if (pointCloud != null)
        {
            SendTransform(
                pointCloud.transform,
                "point_cloud_pose",
                timestamp
            );
        }
    }

    private void SendTransform(Transform target, string type, float timestamp)
    {
        Vector3 pos = target.position;
        Quaternion rot = target.rotation;

        // Convert to OpenCV coordinates if enabled
        if (convertToOpenCV)
        {
            pos = Utils.UnityToOpenCV(pos);
            rot = Utils.UnityToOpenCV(rot);
        }

        PosePayload payload = new PosePayload
        {
            type = type,
            timestamp = timestamp,
            position = Vector3Payload.From(pos),
            rotation = QuaternionPayload.From(rot)
        };

        try
        {
            client.SendJson(payload);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"{LogTag} Write {type} failed ({ex.Message})");
            client.Close();
        }
    }

    private int[] GetViewportSize()
    {
        int[] size = new int[2];
        Payload payload = new Payload
        {
            type = "get_viewport_size",
            timestamp = Time.realtimeSinceStartup
        };

        try
        {
            client.SendJson(payload);
            string response = client.RecvJson();
            Debug.Log($"{LogTag} Raw response: {response}");
            size = ParseJsonIntArray(response, "viewport_size");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"{LogTag} Get viewpoint size failed ({ex.Message})");
        }
        return size;
    }

    private void GetCameraPose(Camera camera, int camIdx = 0)
    {
        CameraPoseRequest request = new CameraPoseRequest
        {
            type = "get_camera_pose",
            cam_idx = camIdx
        };

        try
        {
            client.SendJson(request);
            string response = client.RecvJson();
            Debug.Log($"{LogTag} Raw response: {response}");

            // Server returns: {"position": [x,y,z], "rotation": [x,y,z,w]}
            // JsonUtility can't parse arrays into structs, so parse manually.
            float[] posArr = ParseJsonFloatArray(response, "position");
            float[] rotArr = ParseJsonFloatArray(response, "rotation");

            Vector3 pos = new Vector3(posArr[0], posArr[1], posArr[2]);
            Quaternion rot = new Quaternion(rotArr[0], rotArr[1], rotArr[2], rotArr[3]);

            // Convert from OpenCV back to Unity coordinates if convertToOpenCV is enabled
            if (convertToOpenCV)
            {
                pos = Utils.OpenCVToUnity(pos);
                rot = Utils.OpenCVToUnity(rot);
            }

            camera.transform.position = pos;
            camera.transform.rotation = rot;
            Debug.Log($"{LogTag} Camera pose set: pos=({pos.x}, {pos.y}, {pos.z}), rot=({rot.x}, {rot.y}, {rot.z}, {rot.w})");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"{LogTag} Get camera pose failed ({ex.Message})");
        }
    }

    /// <summary>
    /// Extract a float array from a JSON string by key name.
    /// Handles: "key": [1.0, 2.0, 3.0]
    /// </summary>
    private static float[] ParseJsonFloatArray(string json, string key)
    {
        string search = $"\"{key}\"";
        int keyIdx = json.IndexOf(search);
        if (keyIdx < 0) return Array.Empty<float>();

        int bracketStart = json.IndexOf('[', keyIdx);
        int bracketEnd = json.IndexOf(']', bracketStart);
        string inner = json.Substring(bracketStart + 1, bracketEnd - bracketStart - 1);

        string[] parts = inner.Split(',');
        float[] result = new float[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            result[i] = float.Parse(parts[i].Trim(), System.Globalization.CultureInfo.InvariantCulture);
        }
        return result;
    }

    private static int[] ParseJsonIntArray(string json, string key)
    {
        string search = $"\"{key}\"";
        int keyIdx = json.IndexOf(search);
        if (keyIdx < 0) return Array.Empty<int>();

        int bracketStart = json.IndexOf('[', keyIdx);
        int bracketEnd = json.IndexOf(']', bracketStart);
        string inner = json.Substring(bracketStart + 1, bracketEnd - bracketStart - 1);

        string[] parts = inner.Split(',');
        int[] result = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            result[i] = int.Parse(parts[i].Trim(), System.Globalization.CultureInfo.InvariantCulture);
        }
        return result;
    }

    private void OnDestroy()
    {
        client.Close();
    }
}
