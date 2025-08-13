using Newtonsoft.Json.Linq;
using UnityEngine;

// Worker의 정보를 저장하기 위한 클래스.
public class WorkerInfo
{
	public bool checkbox;				// ?

	public string id = "";				// 작업자 ID
	public string name = "정보 없음";		// 이름
	public string team = "";			// 부서
	public string rank = "";			// 직급
	public string worktime = "";		// 근무시간
	public bool on_offline = false;     // 온오프, 현재 안씀
	public string position = "(0, 0)";

	// JSON 문자열로 변수 설정.
	public void SetInfoFromJSONString(string jsonString)
	{
		JObject jsonObject = JObject.Parse(jsonString);

		id = jsonObject.GetValue("workerID")?.Value<string>();
		name = jsonObject.GetValue("이름")?.Value<string>();
		team = jsonObject.GetValue("부서")?.Value<string>();
		rank = jsonObject.GetValue("직급")?.Value<string>();
		worktime = jsonObject.GetValue("근무시간")?.Value<string>();
	}

	// 현재 위치를 텍스트 형식으로 기록. WorkerController의 Update 함수에서 호출됨. (매 프레임 업데이트)
	public void SetPosition(float x, float y)
	{
		position = $"({x:F1}, {y:F1})";
	}
}
