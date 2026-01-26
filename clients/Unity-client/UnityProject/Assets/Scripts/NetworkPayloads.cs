using System;
using UnityEngine;

[Serializable]
public struct Vector3Payload
{
    public float x;
    public float y;
    public float z;

    public static Vector3Payload From(Vector3 v)
    {
        return new Vector3Payload { x = v.x, y = v.y, z = v.z };
    }
}

[Serializable]
public struct QuaternionPayload
{
    public float x;
    public float y;
    public float z;
    public float w;

    public static QuaternionPayload From(Quaternion q)
    {
        return new QuaternionPayload { x = q.x, y = q.y, z = q.z, w = q.w };
    }
}

[Serializable]
public struct PosePayload
{
    public string type;
    public float timestamp;
    public Vector3Payload position;
    public QuaternionPayload rotation;
}
