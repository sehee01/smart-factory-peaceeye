using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

[System.Serializable]
public class ZoneHistoryData
{
    public string zone_id;
    public string hour;
    public int data_points;
    public float avg_cycle_time;
    public int total_ppe_violations;
    public int total_hazard_dwell;
}

public class ZoneHistoryFetcher : MonoBehaviour
{
    public static ZoneHistoryFetcher Instance;
    public string serverUrl = "http://localhost:5000";  // Node.js 주소

    void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    public IEnumerator FetchZoneHistory(string zoneId)
    {
        string url = $"{serverUrl}/zones/{zoneId}/history";

        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string json = "{\"history\":" + request.downloadHandler.text + "}"; // 배열을 래핑

                ZoneHistoryList wrapper = JsonUtility.FromJson<ZoneHistoryList>(json);
                ZoneHistoryUI.Instance.Display(wrapper.history);
            }
            else
            {
                Debug.LogError($"[ZoneHistoryFetcher] API 요청 실패: {request.error}");
            }
        }
    }

    [System.Serializable]
    public class ZoneHistoryList
    {
        public ZoneHistoryData[] history;
    }
}
