using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using UnityEngine;

public class WorkerController : MonoBehaviour
{
    private Dictionary<string, GameObject> workers = new Dictionary<string, GameObject>();
    GameObject workerInfo;

    // Inspector에서 연결할 WorkerInfoUI
    public WorkerInfoUI workerInfoUI;

    void Awake()
    {
        if (workerInfoUI == null)
        {
            workerInfoUI = Object.FindFirstObjectByType<WorkerInfoUI>();
        }
    }

    void Start()
    {
        // 모든 Worker 오브젝트 수집
        GameObject[] allWorkers = GameObject.FindGameObjectsWithTag("Worker");

        foreach (GameObject worker in allWorkers)
        {
            workers[worker.name] = worker;
            Debug.Log("등록된 작업자: " + worker.name);

            /*// 클릭 이벤트 추가 (옵션: Worker마다 클릭 스크립트 부착 시 생략 가능)
            if (worker.GetComponent<WorkerClickHandler>() == null)
            {
                var clickHandler = worker.AddComponent<WorkerClickHandler>();
                clickHandler.controller = this;
                clickHandler.workerID = worker.name;
            }*/
        }
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))     //worker 프리팹을 마우스 좌클릭 시 panelObject ON
        {
            if (workerInfo != null)
                workerInfo.SetActive(true);
        }
        //if ()
        //{   //상태창의 X 버튼 클릭 시 panelObject OFF
        //    if (workerInfo != null)
        //        workerInfo.SetActive(false);
        //}
    }

    // 위치 업데이트 함수
    public void SetWorkerPosition(string json)
    {
        Debug.Log("JS로부터 받은 데이터: " + json);

        JObject data = JObject.Parse(json);
        string id = data["id"].ToString();
        float x = data["x"].ToObject<float>();
        float y = data["y"].ToObject<float>();

        if (workers.ContainsKey(id))
        {
            GameObject worker = workers[id];
            // 위치 업데이트를 AutoMove에 맡김
            AutoMoveWorker mover = worker.GetComponent<AutoMoveWorker>();
            if (mover != null)
            {
                mover.SetTarget(new Vector3(x, worker.transform.position.y, y));
            }
        }
        else
        {
            Debug.LogWarning($"존재하지 않는 worker ID: {id}");
        }
    }

    // 클릭 시 정보 보여주는 함수  
    public void ShowWorkerInfo(string id)
    {
        Debug.Log("ShowWorkerInfo");

        if (!workers.ContainsKey(id))
            return;
        if (workerInfoUI == null)
        {
            Debug.LogError("workerInfoUI가 할당되지 않았습니다!");
            return;
        }

        GameObject worker = workers[id];
        Vector3 pos = worker.transform.position;

        Debug.Log(id);

        // 여기에서 JSON 문자열로 데이터를 직접 작성하거나 전달받은 값을 쓰세요
        string aiJsonString = @"{
            'cameraID': 'CAM_01',
            'workerID': 'W123',
            'position_x': 10.5,
            'position_y': 8.3
        }";

        string dbJsonString = @"{
            'workerID': 'W123',
            '이름': '홍길동',
            '부서': '생산1팀',
            '직급': '대리',
            '근무시간': '5시간 40분'
        }";

        // JObject로 파싱 후 문자열로 다시 전달
        JObject aiJson = JObject.Parse(aiJsonString);
        JObject dbJson = JObject.Parse(dbJsonString);

        Debug.Log(aiJson);
        Debug.Log(dbJson);

        workerInfoUI.SetWorkerInfoFromJson(aiJsonString, dbJsonString);
    }

}