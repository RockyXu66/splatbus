using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class SplatbusMessageClient
{
    private TcpClient client;
    private NetworkStream stream;

    public bool IsConnected => client != null && client.Connected && stream != null;

    public void Connect(string host, int port)
    {
        client = new TcpClient();
        client.NoDelay = true;
        client.Connect(host, port);
        stream = client.GetStream();
    }

    public void SendJson<T>(T payload)
    {
        if (!IsConnected)
        {
            throw new InvalidOperationException("Client is not connected.");
        }

        string json = JsonUtility.ToJson(payload);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        byte[] lengthPrefix = BitConverter.GetBytes(bytes.Length);

        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(lengthPrefix);
        }

        stream.Write(lengthPrefix, 0, lengthPrefix.Length);
        stream.Write(bytes, 0, bytes.Length);
    }

    public string RecvJson()
    {
        if (!IsConnected)
        {
            throw new InvalidOperationException("Client is not connected.");
        }

        // Read 4-byte length prefix (little-endian)
        byte[] lengthBuf = new byte[4];
        ReadExact(lengthBuf, 4);
        int length = BitConverter.ToInt32(lengthBuf, 0);
        if (!BitConverter.IsLittleEndian)
        {
            length = System.Net.IPAddress.NetworkToHostOrder(length);
        }

        // Read payload
        byte[] payload = new byte[length];
        ReadExact(payload, length);
        return Encoding.UTF8.GetString(payload);
    }

    private void ReadExact(byte[] buffer, int count)
    {
        int offset = 0;
        while (offset < count)
        {
            int read = stream.Read(buffer, offset, count - offset);
            if (read == 0)
            {
                throw new System.IO.EndOfStreamException("Connection closed");
            }
            offset += read;
        }
    }

    public void Close()
    {
        if (stream != null)
        {
            stream.Dispose();
            stream = null;
        }

        if (client != null)
        {
            try
            {
                client.Close();
            }
            catch
            {
                // Ignore socket close errors.
            }
            client = null;
        }
    }
}