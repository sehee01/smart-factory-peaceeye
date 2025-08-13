using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class WorkersManager : MonoBehaviour
{
	private Dictionary<string, GameObject> workers = new Dictionary<string, GameObject>();
	[SerializeField]
	GameObject workerPrefeb;
	static int nextId = 1;

	// Inspector에서 연결할 WorkerInfoUI
	[SerializeField]
	WorkerInfoUI workerInfoUI;

	void Awake()
	{
		if (workerInfoUI == null)
		{
			workerInfoUI = Object.FindFirstObjectByType<WorkerInfoUI>();
		}
	}

	void Start()
	{
		// 시작 시 새 worker 1개 생성
		createNewWorker();
	}

	void Update()
	{
		// 메인 카메라가 활성화 되지 않은 상태면 아무것도 하지 않음
		if (!Camera.main)
		{
			return;
		}

		// 마우스 왼쪽 클릭하면 동작.
		if (Mouse.current.leftButton.wasPressedThisFrame)
		{
			Ray ray = Camera.main.ScreenPointToRay(Mouse.current.position.ReadValue());
			if (Physics.Raycast(ray, out RaycastHit hit))
			{
				//Debug.Log(hit.collider.name + "(" + hit.collider.tag + ")");

				// 클릭한 물체의 tag가 Worker일 때만 정보창 띄우기
				if (hit.collider.tag == "Worker")
					ShowWorkerInfo(hit.collider.name);
			}
		}
	}

	// 새로운 worker 추가
	// Prefeb으로 새 worker 를 생성하고 아이디 부여 및 목록에 추가
	public void createNewWorker()
	{
		// 프리펩이 지정되어있는지 검사. worker 프리펩을 지정해야 함.
		if (!workerPrefeb)
		{
			Debug.Log("Worker controller에 Worker prefeb이 지정되어 있지 않음.");
			return;
		}

		// 프리펩으로 새 worker 생성
		GameObject worker = GameObject.Instantiate(workerPrefeb);

		// 새 worker의 아이디 지정. nextId는 worker가 새로 생성될 때마다 증가.
		worker.name = "Worker" + nextId++;

		// workers 디렉토리에 새 워커 추가.
		workers[worker.name] = worker;
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
		//Debug.Log("ShowWorkerInfo");

		// workders에 해당 이름의 worker가 있는지 검사. 없으면 안됨.
		if (!workers.ContainsKey(id))
		{
			Debug.LogError("해당 아이디를 가진 worker가 목록에 없음 : " + id);
			return;
		}

		// worker 정보 보여주는 UI object가 지정되어 있는지 검사.
		if (workerInfoUI == null)
		{
			Debug.LogError("workerInfoUI가 할당되지 않았습니다!");
			return;
		}

		// 해당 이름을 가진 worker를 리스트에서 가져옴.
		GameObject worker = workers[id];
		Vector3 pos = worker.transform.position;

		// worker 정보창 띄움.
		workerInfoUI.SetWorkerInfo(worker);

		//Debug.Log(id);

		//// 여기에서 JSON 문자열로 데이터를 직접 작성하거나 전달받은 값을 쓰세요
		//string aiJsonString = @"{
		//          'cameraID': 'CAM_01',
		//          'workerID': 'W123',
		//          'position_x': 10.5,
		//          'position_y': 8.3
		//      }";

		//string dbJsonString = @"{
		//          'workerID': 'W123',
		//          '이름': '홍길동',
		//          '부서': '생산1팀',
		//          '직급': '대리',
		//          '근무시간': '5시간 40분'
		//      }";

		//// JObject로 파싱 후 문자열로 다시 전달
		//JObject aiJson = JObject.Parse(aiJsonString);
		//JObject dbJson = JObject.Parse(dbJsonString);

		//Debug.Log(aiJson);
		//Debug.Log(dbJson);

		//workerInfoUI.SetWorkerInfoFromJson(aiJsonString, dbJsonString);
	}

	Dictionary<string, GameObject> GetWorkers() { return workers; }
}