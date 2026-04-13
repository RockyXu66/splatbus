using UnityEngine;

public static class Utils
{
    /// <summary>
    /// Convert Unity position to OpenCV coordinate system (flip Y axis).
    /// Unity: X-right, Y-up, Z-forward
    /// OpenCV: X-right, Y-down, Z-forward
    /// </summary>
    public static Vector3 UnityToOpenCV(Vector3 unityPos)
    {
        return new Vector3(unityPos.x, -unityPos.y, unityPos.z);
    }

    /// <summary>
    /// Convert Unity rotation to OpenCV coordinate system (flip Y axis).
    /// When Y axis is flipped, the quaternion transform is: (x, y, z, w) → (-x, y, -z, w)
    /// </summary>
    public static Quaternion UnityToOpenCV(Quaternion unityRot)
    {
        return new Quaternion(-unityRot.x, unityRot.y, -unityRot.z, unityRot.w);
    }

    /// <summary>
    /// Convert OpenCV position to Unity coordinate system (flip Y axis).
    /// </summary>
    public static Vector3 OpenCVToUnity(Vector3 cvPos)
    {
        return new Vector3(cvPos.x, -cvPos.y, cvPos.z);
    }

    /// <summary>
    /// Convert OpenCV quaternion to Unity quaternion (flip Y axis).
    /// When Y axis is flipped, the quaternion transform is: (x, y, z, w) → (-x, y, -z, w)
    /// </summary>
    public static Quaternion OpenCVToUnity(Quaternion cvRot)
    {
        return new Quaternion(-cvRot.x, cvRot.y, -cvRot.z, cvRot.w);
    }
}
