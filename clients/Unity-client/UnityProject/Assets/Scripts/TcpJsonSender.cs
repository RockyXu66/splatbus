using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class TcpJsonSender
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
            throw new InvalidOperationException("Sender is not connected.");
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