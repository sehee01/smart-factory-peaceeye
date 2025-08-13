using Newtonsoft.Json.Linq;
using TMPro;
using UnityEngine;
using static WorkerTableUI;

public class WorkerInfoUI : MonoBehaviour
{
	[Header("UI Components")]
	public TextMeshProUGUI workerID;
	public TextMeshProUGUI workerName;
	public TextMeshProUGUI workerTeam;
	public TextMeshProUGUI workerRank;
	public TextMeshProUGUI workTimeText;
	public TextMeshProUGUI positionText;

	[Header("Cameras")]
	//public Camera workerCamera;
	public Camera overviewCamera;

	private string currentWorkerID;     // 추가 정보 요청용
	private GameObject currentWorker;   // 현재 선택된 worker

	[SerializeField]
	UIPageCanvasController canvasController;


	void Start()
	{
		gameObject.SetActive(false);
	}

	void Update()
	{
		// 실시간으로 worker의 위치로 갱신.
		UpdateWorkerPosition();
	}


	// worker를 입력받고, worker의 정보를 화면에 표시
	public void SetWorkerInfo(GameObject worker)
	{
		// 정보창 활성화
		//gameObject.SetActive(true);
		canvasController.ShowWorkerInfo();

		// worker가 있는지 확인
		if (!worker)
		{
			Debug.LogWarning("해당 worker가 없습니다.");
			return;
		}

		// 현재의 worker에 입력된 worker 넣음.
		currentWorker = worker;

		// 입력된 worker의 정보로 UI를 채움.
		WorkerController workerController = worker.GetComponent<WorkerController>();
		//Debug.Log(workerController);
		workerID.text = workerController.workerInfo.id;
		workerName.text = workerController.workerInfo.name;
		workerTeam.text = workerController.workerInfo.team;
		workerRank.text = workerController.workerInfo.rank;
		workTimeText.text = workerController.workerInfo.worktime;

		// 현재 worker의 ID 기록. 추가 정보 요청때 사용.
		currentWorkerID = workerController.workerInfo.id;

		// worker의 위치는 실시간으로 변경되는 것을 반영하기 위해 Update에서 기록.
		//// worker의 위치 읽음.
		//positionText.text = $"({x:F1}, {y:F1})";

		// 전경 카메라 켜기, 기본 카메라 끄기
		if (overviewCamera != null) overviewCamera.gameObject.SetActive(false);
		if (currentWorker != null) currentWorker.GetComponent<WorkerController>().SetActiveTopviewCamera(true);
	}

	
	// worker의 위치를 positionText에 업데이트. 실시간 반영을 위해 Update 함수에서만 호출.
	void UpdateWorkerPosition()
	{
		// 현재 선택된 worker가 없으면 업데이트 안함.
		if(!currentWorker)
		{
			return;
		}

		positionText.text = currentWorker.GetComponent<WorkerController>().workerInfo.position;
		//float x = currentWorker.transform.position.x;
		//float z = currentWorker.transform.position.z;
		//positionText.text = $"({x:F1}, {z:F1})";
	}

	//// JSON으로 패널 열고 내용 설정
	//public void SetWorkerInfoFromJson(string aiJson, string dbJson)
	//{
	//	Debug.Log("SetWorkerInfoFromJson");
	//	// 패널 표시
	//	gameObject.SetActive(true);

	//	if (string.IsNullOrEmpty(aiJson) || string.IsNullOrEmpty(dbJson))
	//	{
	//		Debug.LogWarning("JSON 입력이 비어 있습니다.");
	//		return;
	//	}

	//	JObject aiData = JObject.Parse(aiJson);
	//	JObject dbData = JObject.Parse(dbJson);

	//	string aiWorkerID = aiData["workerID"]?.ToString();
	//	string dbWorkerID = dbData["workerID"]?.ToString();

	//	if (aiWorkerID != dbWorkerID)
	//	{
	//		Debug.LogWarning("AI 데이터와 DB 데이터의 workerID가 일치하지 않습니다.");
	//		return;
	//	}

	//	currentWorkerID = aiWorkerID;

	//	// UI 업데이트
	//	workerID.text = $"{aiWorkerID}";
	//	workerName.text = $"{dbData["이름"]}";
	//	workerTeam.text = $"{dbData["부서"]}";
	//	workerRank.text = $"{dbData["직급"]}";
	//	workTimeText.text = $"{dbData["근무시간"]}";

	//	float x = aiData["position_x"]?.ToObject<float>() ?? 0f;
	//	float y = aiData["position_y"]?.ToObject<float>() ?? 0f;
	//	positionText.text = $"({x:F1}, {y:F1})";

	//	// 전경 카메라 켜기, 기본 카메라 끄기
	//	if (overviewCamera != null) overviewCamera.gameObject.SetActive(false);
	//	if (workerCamera != null) workerCamera.gameObject.SetActive(true);
	//}

	// X 버튼 누르면 닫기 요청
	public void HidePanel()
	{
		//gameObject.SetActive(false);

		// 메인 카메라 복귀
		if (overviewCamera != null) overviewCamera.gameObject.SetActive(true);
		if (currentWorker != null) currentWorker.GetComponent<WorkerController>().SetActiveTopviewCamera(false);
	}

	// 추가 정보 요청 (버튼 연결)
	public void RequestMoreInfo()
	{
		Debug.Log("추가 정보 요청: " + currentWorkerID);

		// JS 또는 서버 요청 호출 예시
#if UNITY_WEBGL && !UNITY_EDITOR
        //Application.ExternalCall("RequestWorkerDetail", currentWorkerID);
#else
		// 또는 유니티 내부에서 서버 호출하는 방식
		// StartCoroutine(RequestInfoFromServer(currentWorkerID));
#endif
	}
}