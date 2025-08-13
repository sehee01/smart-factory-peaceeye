using Newtonsoft.Json.Linq;
using Unity.VisualScripting;
using UnityEngine;


[RequireComponent(typeof(Rigidbody))]
public class WorkerController : MonoBehaviour
{
	// worker의 정보 기록. JSON으로 파싱해서 넣을 수 있음.
	public WorkerInfo workerInfo = new WorkerInfo();

	// 자동으로 움직이기 위한 변수들.
	public float moveSpeed = 1f;
	public float rotationSpeed = 5f;
	public float arriveThreshold = 0.1f;

	// 위에서 내려다보는 카메라
	GameObject workerCamera;


	// Start is called once before the first execution of Update after the MonoBehaviour is created
	void Start()
    {
		// 임시 데이터. 삭제해야 함.
		// 여기에서 JSON 문자열로 데이터를 직접 작성하거나 전달받은 값을 쓰세요
		//string aiJsonString = @"{
		//          'cameraID': 'CAM_01',
		//          'workerID': 'W123',
		//          'position_x': 10.5,
		//          'position_y': 8.3
		//      }";

		string dbJsonString = @"{
            'workerID': 'W123',
            '이름': '홍길동',
            '부서': '생산1팀',
            '직급': '대리',
            '근무시간': '5시간 40분'
        }";


		//// JObject로 파싱 후 문자열로 다시 전달
		//JObject aiJson = JObject.Parse(aiJsonString);
		//JObject dbJson = JObject.Parse(dbJsonString);
		//Debug.Log(aiJson);
		//Debug.Log(dbJson);


		// JSON 문자열로 worker의 정보 설정 (임시, 후에 서버에서 받아온 JSON 문자열을 넣게 교체해야함.)
		workerInfo.SetInfoFromJSONString(dbJsonString);
        //StartCoroutine(GetWorkerDataFromServer());

        // 하위 object 중에 workerCamera를 찾음.
        workerCamera = transform.Find("workerCamera")?.gameObject;
	}

	// worker camera를 활성화 / 비활성화
	public void SetActiveTopviewCamera(bool enable)
	{
		// 카메라가 없으면 동작 안함
		if(!workerCamera)
		{
			Debug.LogWarning("하위 오브젝트 중에 카메라가 없습니다.");
			return;
		}

		// 활성화 / 비활성화
		workerCamera.SetActive(enable);
	}

    // Update is called once per frame
    void Update()
    {
		// 현재 위치를 workerInfo에 텍스트 형식으로 계속 업데이트
		workerInfo.SetPosition(transform.position.x, transform.position.z);
    }
}
