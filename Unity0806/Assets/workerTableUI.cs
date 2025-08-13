using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class WorkerTableUI : MonoBehaviour
{
    //[System.Serializable]
    //public class Worker
    //{
    //    public bool checkbox;
    //    public string name;
    //    public string team;
    //    public string position;
    //    public string worktime; //datatime을 string형으로
    //    public bool on_offline;
    //}

    [SerializeField]
    WorkersManager workersManager;
    public RectTransform contentParent; // Content 오브젝트 참조
    public GameObject rowPrefab;        // 프리팹: 한 줄(Row)

    private List<WorkerInfo> workers = new List<WorkerInfo>();

    void Start()
    {
        // 예시용 작업자 목록
        workers.Add(new WorkerInfo { checkbox = true, name = "김철수", team = "홍보팀", rank = "팀원", position = "(100, 200)", worktime = "5:30", on_offline = true });
        //x: 100, y: 200
        //5시간 30분
        workers.Add(new WorkerInfo { checkbox = false, name = "안영희", team = "개발팀", rank = "팀원", position = "(0, 0)", worktime = "0:00", on_offline = false });
        workers.Add(new WorkerInfo { checkbox = false, name = "박수인", team = "개발팀", rank = "팀원", position = "(10, 50)", worktime = "1:00", on_offline = true });
        PopulateTable();
    }

    public void PopulateTable()
    {
        foreach (Transform child in contentParent)
        {
            Debug.Log("Destroy");
            //Destroy(child.gameObject); // 기존 행 제거
        }

        foreach (var worker in workers)
        {
            GameObject row = Instantiate(rowPrefab, contentParent);
            TextMeshProUGUI[] texts = row.GetComponentsInChildren<TextMeshProUGUI>();

            texts[0].text = worker.name;
            texts[1].text = worker.team;
            texts[2].text = worker.position;
            texts[3].text = worker.worktime;
            //texts[4].text = worker.on_offline;
        }
    }

    //public void AddWorker(string name, string team, string position, string worktime, bool on_offline)
    //{
    //    workers.Add(new WorkerInfo { name = name, team = team, position = position, worktime = worktime, on_offline = on_offline });
    //    PopulateTable();
    //}

    // worker 오브젝트를 리스트에 추가
    public void AddWorker(GameObject worker)
	{
        workers.Add(worker.GetComponent<WorkerController>().workerInfo);
		PopulateTable();
	}
}
