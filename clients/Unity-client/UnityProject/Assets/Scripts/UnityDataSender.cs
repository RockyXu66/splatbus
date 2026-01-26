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
    private TcpJsonSender sender;

    private const string LogTag = "[UnityDataSender]";

    private void Awake()
    {
        sender = new TcpJsonSender();
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
    }

    private void Update()
    {
        if (!sender.IsConnected)
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
        sender.Close();

        try
        {
            sender.Connect(host, port);
            Debug.Log($"{LogTag} Connected to {host}:{port}");
        }
        catch (Exception)
        {
            // Debug.LogWarning($"{LogTag} Failed to connect to {host}:{port} ({ex.Message})");
            sender.Close();
        }
    }

    private void SendCameraPose()
    {
        if (!sender.IsConnected)
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
        if (!sender.IsConnected)
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
            sender.SendJson(payload);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"{LogTag} Write {type} failed ({ex.Message})");
            sender.Close();
        }
    }

    private void OnDestroy()
    {
        sender.Close();
    }
}
