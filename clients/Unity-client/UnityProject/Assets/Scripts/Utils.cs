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
}
