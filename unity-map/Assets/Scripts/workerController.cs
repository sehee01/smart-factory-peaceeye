using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json.Linq; // Unity에 JSON 파싱을 쉽게 하려면 JSON.NET 권장

public class WorkerController : MonoBehaviour
{
    private Dictionary<string, GameObject> workers = new Dictionary<string, GameObject>();

    void Start()
    {
        // 오브젝트를 자동으로 찾아서 딕셔너리에 저장
        GameObject[] allWorkers = GameObject.FindGameObjectsWithTag("Worker");

        foreach (GameObject worker in allWorkers)
        {
            workers[worker.name] = worker;
            Debug.Log("📦 등록된 작업자: " + worker.name);
        }
    }

    public void SetWorkerPosition(string json)
    {
        Debug.Log("📩 JS로부터 받은 데이터: " + json);

        JObject data = JObject.Parse(json);
        string id = data["id"].ToString();
        float x = data["x"].ToObject<float>();
        float y = data["y"].ToObject<float>();

        if (workers.ContainsKey(id))
        {
            GameObject worker = workers[id];
            worker.transform.position = new Vector3(x, worker.transform.position.y, y); // y축 고정, 2D면 z 고정
        }
        else
        {
            Debug.LogWarning($"🚫 존재하지 않는 worker ID: {id}");
        }
    }
}
