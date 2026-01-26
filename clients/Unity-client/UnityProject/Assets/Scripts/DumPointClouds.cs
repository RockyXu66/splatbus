
using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

public class DumPointClouds : MonoBehaviour
{
    [Header("PLY File Settings")]
    [Tooltip("Path to the binary PLY file")]
    public string plyFilePath = "";
    
    [Tooltip("Load PLY file on start")]
    public bool loadOnStart = true;
    
    [Header("Display Settings")]
    [Tooltip("Point size in world units")]
    [Range(0.001f, 0.1f)]
    [HideInInspector] private float pointSize = 0.01f;
    public bool showInGameView = false;
    
    [Tooltip("Material for rendering points (leave null for default)")]
    [HideInInspector] private Material pointMaterial;
    
    [Header("Coordinate Conversion")]
    [Tooltip("Convert OpenCV/COLMAP coordinates (X-right, Y-down, Z-forward) to Unity space")]
    public bool convertFromOpenCVAxes = true;
    
    [Header("Debug Info")]
    [SerializeField] private int totalPoints = 0;
    [SerializeField] private int meshCount = 0;
    [HideInInspector] private bool isLoaded = false;
    public bool showDebugGUI = false;
    
    private List<GameObject> meshObjects = new List<GameObject>();
    private const int MAX_VERTICES_PER_MESH = 65000; // Unity's vertex limit
    private int lastAppliedLayer = -1;
    
    // PLY header properties
    private class PlyProperty
    {
        public string name;
        public string type;
        public int bytesSize;
    }
    
    private class PlyHeader
    {
        public int vertexCount;
        public List<PlyProperty> properties = new List<PlyProperty>();
        public bool isBinary;
        public bool isBigEndian;
    }
    
    void Start()
    {
        if (loadOnStart && !string.IsNullOrEmpty(plyFilePath))
        {
            LoadPlyFile(plyFilePath);
        }
    }

    void OnEnable()
    {
        ApplyPointCloudLayerToMeshes();
    }

    void Update()
    {
        if (!Application.isPlaying)
        {
            return;
        }

        int targetLayer = GetPointCloudLayer();
        if (targetLayer != lastAppliedLayer)
        {
            ApplyPointCloudLayerToMeshes();
        }
    }
    
    /// <summary>
    /// Load and display a PLY point cloud file
    /// </summary>
    public void LoadPlyFile(string filePath)
    {
        // Clear previous point clouds
        ClearPointClouds();
        
        if (!File.Exists(filePath))
        {
            Debug.LogError($"[DumPointClouds] File not found: {filePath}");
            return;
        }
        
        Debug.Log($"[DumPointClouds] Loading PLY file: {filePath}");
        
        try
        {
            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader reader = new BinaryReader(fs))
            {
                // Parse header
                PlyHeader header = ParsePlyHeader(reader);
                
                if (!header.isBinary)
                {
                    Debug.LogError("[DumPointClouds] Only binary PLY files are supported!");
                    return;
                }
                
                Debug.Log($"[DumPointClouds] Found {header.vertexCount} vertices");
                totalPoints = header.vertexCount;
                
                // Read vertex data
                List<Vector3> positions = new List<Vector3>();
                List<Color> colors = new List<Color>();
                
                ReadVertexData(reader, header, positions, colors);
                
                // Create meshes to display the points
                CreatePointCloudMeshes(positions, colors);
                
                isLoaded = true;
                Debug.Log($"[DumPointClouds] Successfully loaded {totalPoints} points in {meshCount} mesh(es)");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[DumPointClouds] Error loading PLY file: {e.Message}\n{e.StackTrace}");
        }
    }
    
    /// <summary>
    /// Parse the PLY header to understand the file structure
    /// </summary>
    private PlyHeader ParsePlyHeader(BinaryReader reader)
    {
        PlyHeader header = new PlyHeader();
        string line;
        bool inHeader = true;
        
        // Read ASCII header
        StringBuilder lineBuilder = new StringBuilder();
        
        while (inHeader)
        {
            lineBuilder.Clear();
            char c;
            
            // Read line character by character
            while ((c = reader.ReadChar()) != '\n')
            {
                if (c != '\r')
                    lineBuilder.Append(c);
            }
            
            line = lineBuilder.ToString().Trim();
            
            if (line.StartsWith("ply"))
            {
                continue;
            }
            else if (line.StartsWith("format"))
            {
                string[] tokens = line.Split(' ');
                if (tokens.Length >= 2)
                {
                    header.isBinary = tokens[1].Contains("binary");
                    header.isBigEndian = tokens[1].Contains("big_endian");
                }
            }
            else if (line.StartsWith("element vertex"))
            {
                string[] tokens = line.Split(' ');
                if (tokens.Length >= 3)
                {
                    header.vertexCount = int.Parse(tokens[2]);
                }
            }
            else if (line.StartsWith("property"))
            {
                string[] tokens = line.Split(' ');
                if (tokens.Length >= 3)
                {
                    PlyProperty prop = new PlyProperty();
                    prop.type = tokens[1];
                    prop.name = tokens[2];
                    prop.bytesSize = GetPropertySize(prop.type);
                    header.properties.Add(prop);
                }
            }
            else if (line.StartsWith("end_header"))
            {
                inHeader = false;
            }
        }
        
        return header;
    }
    
    /// <summary>
    /// Get the byte size of a PLY property type
    /// </summary>
    private int GetPropertySize(string type)
    {
        switch (type.ToLower())
        {
            case "char":
            case "uchar":
            case "int8":
            case "uint8":
                return 1;
            case "short":
            case "ushort":
            case "int16":
            case "uint16":
                return 2;
            case "int":
            case "uint":
            case "float":
            case "int32":
            case "uint32":
            case "float32":
                return 4;
            case "double":
            case "float64":
                return 8;
            default:
                return 4;
        }
    }
    
    /// <summary>
    /// Read vertex data from the binary PLY file
    /// </summary>
    private void ReadVertexData(BinaryReader reader, PlyHeader header, List<Vector3> positions, List<Color> colors)
    {
        // Find property indices
        int xIndex = -1, yIndex = -1, zIndex = -1;
        int rIndex = -1, gIndex = -1, bIndex = -1, aIndex = -1;
        
        for (int i = 0; i < header.properties.Count; i++)
        {
            string propName = header.properties[i].name.ToLower();
            if (propName == "x") xIndex = i;
            else if (propName == "y") yIndex = i;
            else if (propName == "z") zIndex = i;
            else if (propName == "red" || propName == "r") rIndex = i;
            else if (propName == "green" || propName == "g") gIndex = i;
            else if (propName == "blue" || propName == "b") bIndex = i;
            else if (propName == "alpha" || propName == "a") aIndex = i;
        }
        
        bool hasColor = (rIndex >= 0 && gIndex >= 0 && bIndex >= 0);
        
        // Read each vertex
        for (int v = 0; v < header.vertexCount; v++)
        {
            Vector3 pos = Vector3.zero;
            Color col = Color.white;
            
            // Read all properties for this vertex
            for (int p = 0; p < header.properties.Count; p++)
            {
                PlyProperty prop = header.properties[p];
                float value = ReadPropertyValue(reader, prop.type);
                
                // Assign to position
                if (p == xIndex) pos.x = value;
                else if (p == yIndex) pos.y = value;
                else if (p == zIndex) pos.z = value;
                // Assign to color
                else if (p == rIndex) col.r = value / 255.0f;
                else if (p == gIndex) col.g = value / 255.0f;
                else if (p == bIndex) col.b = value / 255.0f;
                else if (p == aIndex) col.a = value / 255.0f;
            }
            
            positions.Add(ConvertPointToUnitySpace(pos));
            colors.Add(hasColor ? col : Color.white);
        }
    }
    
    /// <summary>
    /// Convert a vertex from OpenCV/COLMAP coordinates into Unity space
    /// </summary>
    private Vector3 ConvertPointToUnitySpace(Vector3 point)
    {
        Vector3 converted = point;
        
        if (convertFromOpenCVAxes)
        {
            // Flip Y axis
            converted.y = -converted.y;
        }
        
        return converted;
    }
    
    /// <summary>
    /// Read a single property value from the binary stream
    /// </summary>
    private float ReadPropertyValue(BinaryReader reader, string type)
    {
        switch (type.ToLower())
        {
            case "char":
            case "int8":
                return reader.ReadSByte();
            case "uchar":
            case "uint8":
                return reader.ReadByte();
            case "short":
            case "int16":
                return reader.ReadInt16();
            case "ushort":
            case "uint16":
                return reader.ReadUInt16();
            case "int":
            case "int32":
                return reader.ReadInt32();
            case "uint":
            case "uint32":
                return reader.ReadUInt32();
            case "float":
            case "float32":
                return reader.ReadSingle();
            case "double":
            case "float64":
                return (float)reader.ReadDouble();
            default:
                return reader.ReadSingle();
        }
    }
    
    /// <summary>
    /// Create Unity meshes to display the point cloud
    /// </summary>
    private void CreatePointCloudMeshes(List<Vector3> positions, List<Color> colors)
    {
        int totalVertices = positions.Count;
        int numMeshes = Mathf.CeilToInt((float)totalVertices / MAX_VERTICES_PER_MESH);
        meshCount = numMeshes;

        // Keep visible in Scene view but hidden in Game view (only while playing).
        int editorLayer = GetPointCloudLayer();
        
        for (int m = 0; m < numMeshes; m++)
        {
            int startIdx = m * MAX_VERTICES_PER_MESH;
            int count = Mathf.Min(MAX_VERTICES_PER_MESH, totalVertices - startIdx);
            
            // Create mesh
            Mesh mesh = new Mesh();
            mesh.name = $"PointCloud_Mesh_{m}";
            
            // Get vertices and colors for this mesh
            Vector3[] vertices = new Vector3[count];
            Color[] meshColors = new Color[count];
            int[] indices = new int[count];
            
            for (int i = 0; i < count; i++)
            {
                vertices[i] = positions[startIdx + i];
                meshColors[i] = colors[startIdx + i];
                indices[i] = i;
            }
            
            mesh.vertices = vertices;
            mesh.colors = meshColors;
            mesh.SetIndices(indices, MeshTopology.Points, 0);
            mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10000f); // Large bounds
            
            // Create GameObject for this mesh
            GameObject meshObj = new GameObject($"PointCloudMesh_{m}");
            meshObj.transform.SetParent(transform);
            meshObj.transform.localPosition = Vector3.zero;
            meshObj.transform.localRotation = Quaternion.identity;
            meshObj.transform.localScale = Vector3.one;
            meshObj.layer = editorLayer;   // Set visibility
            
            // Add mesh filter and renderer
            MeshFilter filter = meshObj.AddComponent<MeshFilter>();
            filter.mesh = mesh;
            
            MeshRenderer renderer = meshObj.AddComponent<MeshRenderer>();
            
            // Setup material
            if (pointMaterial != null)
            {
                renderer.material = pointMaterial;
            }
            else
            {
                // Create default point cloud material
                Material mat = new Material(Shader.Find("Particles/Standard Unlit"));
                mat.SetFloat("_Mode", 0);
                mat.EnableKeyword("_EMISSION");
                renderer.material = mat;
            }
            
            meshObjects.Add(meshObj);
        }
        
        Debug.Log($"[DumPointClouds] Created {numMeshes} mesh object(s)");
    }
    
    /// <summary>
    /// Clear all loaded point cloud meshes
    /// </summary>
    public void ClearPointClouds()
    {
        foreach (GameObject obj in meshObjects)
        {
            if (obj != null)
            {
                if (Application.isPlaying)
                    Destroy(obj);
                else
                    DestroyImmediate(obj);
            }
        }
        
        meshObjects.Clear();
        totalPoints = 0;
        meshCount = 0;
        isLoaded = false;
    }
    
    void OnDestroy()
    {
        ClearPointClouds();
    }
    
    void OnValidate()
    {
        // Update point size for all materials when changed in inspector
        if (Application.isPlaying && meshObjects.Count > 0)
        {
            foreach (GameObject obj in meshObjects)
            {
                if (obj != null)
                {
                    MeshRenderer renderer = obj.GetComponent<MeshRenderer>();
                    if (renderer != null && renderer.material != null)
                    {
                        renderer.material.SetFloat("_PointSize", pointSize * 100f);
                    }
                }
            }
        }
    }

    private int GetPointCloudLayer()
    {
        int editorOnlyLayer = LayerMask.NameToLayer("EditorOnly");
        if (Application.isPlaying && !showInGameView && editorOnlyLayer != -1)
        {
            return editorOnlyLayer;
        }

        return LayerMask.NameToLayer("Default");
    }

    private void ApplyPointCloudLayerToMeshes()
    {
        int layer = GetPointCloudLayer();
        if (meshObjects.Count == 0)
        {
            lastAppliedLayer = layer;
            return;
        }

        foreach (GameObject obj in meshObjects)
        {
            if (obj != null)
            {
                obj.layer = layer;
            }
        }
        lastAppliedLayer = layer;
    }
    
    #region Debug GUI
    void OnGUI()
    {
        if (!showDebugGUI) return;

        if (isLoaded)
        {
            GUILayout.BeginArea(new Rect(Screen.width - 260, 10, 250, 150));
            GUILayout.Box("=== Point Cloud Info ===");
            GUILayout.Label($"Total Points: {totalPoints:N0}");
            GUILayout.Label($"Mesh Count: {meshCount}");
            GUILayout.Label($"File: {Path.GetFileName(plyFilePath)}");
            
            if (GUILayout.Button("Clear Point Cloud"))
            {
                ClearPointClouds();
            }
            
            GUILayout.EndArea();
        }
    }
    #endregion
}
