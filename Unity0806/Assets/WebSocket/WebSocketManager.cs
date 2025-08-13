using UnityEngine;
using NativeWebSocket;
using System;
using System.Text;

#region Data Models

[System.Serializable]
public class Worker
{
    public string worker_id;
    public float x;
    public float y;
    public string zone_id;
    public int product_count;
    public float cycle_time_min;
    public string timestamp;
}

[System.Serializable]
public class ZoneRealtimeData
{
    public string zone_id;
    public string zone_name;
    public string zone_type;
    public int active_workers;
    public string active_tasks;
    public float avg_cycle_time_min;
    public int ppe_violations;
    public int hazard_dwell_count;
    public string recent_alerts;
}

[System.Serializable]
public class ViolationDetail
{
    public string[] ppe;
    public string[] roi;
}

[System.Serializable]
public class ViolationEntry
{
    public string worker_id;
    public string zone_id;
    public string timestamp;
    public ViolationDetail violations;
}

#endregion

#region WebSocket Wrapper

[System.Serializable]
public class WorkersPayload
{
    public string timestamp;
    public Worker[] workers;
}

[System.Serializable]
public class ZonesPayload
{
    public string timestamp;
    public ZoneRealtimeData[] zones;
}

[System.Serializable]
public class ViolationsPayload
{
    public string timestamp;
    public ViolationEntry[] violations;
}

[System.Serializable]
public class WebSocketWrapper
{
    public string type;
    public string payload;
}

#endregion

public class WebSocketManager : MonoBehaviour
{
    public static WebSocketManager Instance { get; private set; }

    public event Action<Worker[]> OnWorkersReceived;
    public event Action<ZoneRealtimeData[]> OnZonesReceived;
    public event Action<ViolationEntry[]> OnViolationsReceived;

    private WebSocket websocket;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject); // 중복 제거
        }
    }

    async void Start()
    {
        websocket = new WebSocket("ws://localhost:8000");

        websocket.OnOpen += () => Debug.Log("[WebSocketManager] 연결됨");
        websocket.OnError += (e) => Debug.LogError("[WebSocketManager] 오류: " + e);

        websocket.OnMessage += (bytes) =>
        {
            string json = Encoding.UTF8.GetString(bytes);
            Debug.Log("[WebSocketManager] 수신된 메시지: " + json);

            try
            {
                // 전체 메시지에서 type 추출
                string type = ExtractValue(json, "type");

                // payload 부분만 추출
                string payloadJson = ExtractPayload(json);

                if (string.IsNullOrEmpty(type) || string.IsNullOrEmpty(payloadJson))
                {
                    Debug.LogError("[WebSocketManager] type 또는 payload가 null");
                    return;
                }

                switch (type)
                {
                    case "workers_update":
                        var workersPayload = JsonUtility.FromJson<WorkersPayload>(payloadJson);
                        if (workersPayload?.workers != null)
                            OnWorkersReceived?.Invoke(workersPayload.workers);
                        break;

                    case "zone_update":
                        var zonesPayload = JsonUtility.FromJson<ZonesPayload>(payloadJson);
                        if (zonesPayload?.zones != null)
                            OnZonesReceived?.Invoke(zonesPayload.zones);
                        break;

                    case "violation_update":
                        var violationsPayload = JsonUtility.FromJson<ViolationsPayload>(payloadJson);
                        if (violationsPayload?.violations != null)
                            OnViolationsReceived?.Invoke(violationsPayload.violations);
                        break;

                    default:
                        Debug.LogWarning("[WebSocketManager] 알 수 없는 type: " + type);
                        break;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError("[WebSocketManager] JSON 파싱 오류: " + ex.Message);
            }
        };

        await websocket.Connect();
        Debug.Log("[WebSocketManager] WebSocket 연결 완료");
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
#endif
    }

    async void OnApplicationQuit()
    {
        await websocket?.Close();
    }

    private string ExtractValue(string json, string key)
    {
        string search = $"\"{key}\":\"";
        int startIndex = json.IndexOf(search) + search.Length;
        if (startIndex < search.Length) return null;

        int endIndex = json.IndexOf("\"", startIndex);
        return json.Substring(startIndex, endIndex - startIndex);
    }

    private string ExtractPayload(string json)
    {
        int payloadStart = json.IndexOf("\"payload\":") + "\"payload\":".Length;
        if (payloadStart < "\"payload\":".Length) return null;

        int braceCount = 0;
        int startIndex = -1, endIndex = -1;
        for (int i = payloadStart; i < json.Length; i++)
        {
            if (json[i] == '{')
            {
                if (braceCount == 0) startIndex = i;
                braceCount++;
            }
            else if (json[i] == '}')
            {
                braceCount--;
                if (braceCount == 0)
                {
                    endIndex = i;
                    break;
                }
            }
        }

        if (startIndex != -1 && endIndex != -1)
        {
            return json.Substring(startIndex, endIndex - startIndex + 1);
        }

        return null;
    }
}
