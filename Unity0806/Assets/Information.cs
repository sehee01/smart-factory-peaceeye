using Newtonsoft.Json.Linq;
using TMPro;
using UnityEngine;

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
    public Camera workerCamera;
    public Camera overviewCamera;

    private string currentWorkerID;     // 추가 정보 요청용

    void Start()
    {
        gameObject.SetActive(false);
    }

    // JSON으로 패널 열고 내용 설정
    public void SetWorkerInfoFromJson(string aiJson, string dbJson)
    {
        Debug.Log("SetWorkerInfoFromJson");
        // 패널 표시
        gameObject.SetActive(true);

        if (string.IsNullOrEmpty(aiJson) || string.IsNullOrEmpty(dbJson))
        {
            Debug.LogWarning("JSON 입력이 비어 있습니다.");
            return;
        }

        JObject aiData = JObject.Parse(aiJson);
        JObject dbData = JObject.Parse(dbJson);

        string aiWorkerID = aiData["workerID"]?.ToString();
        string dbWorkerID = dbData["workerID"]?.ToString();

        if (aiWorkerID != dbWorkerID)
        {
            Debug.LogWarning("AI 데이터와 DB 데이터의 workerID가 일치하지 않습니다.");
            return;
        }

        currentWorkerID = aiWorkerID;

        // UI 업데이트
        workerID.text = $"ID: {aiWorkerID}";
        workerName.text = $"이름: {dbData["이름"]}";
        workerTeam.text = $"부서: {dbData["부서"]}";
        workerRank.text = $"직급: {dbData["직급"]}";
        workTimeText.text = $"근로시간: {dbData["근무시간"]}";

        float x = aiData["position_x"]?.ToObject<float>() ?? 0f;
        float y = aiData["position_y"]?.ToObject<float>() ?? 0f;
        positionText.text = $"현재 위치: ({x:F1}, {y:F1})";

        // 전경 카메라 켜기, 기본 카메라 끄기
        if (overviewCamera != null) overviewCamera.gameObject.SetActive(true);
        if (workerCamera != null) workerCamera.gameObject.SetActive(false);
    }

   // X 버튼 누르면 닫기 요청
    public void HidePanel()
    {
        gameObject.SetActive(false);

        // 메인 카메라 복귀
        if (workerCamera != null) overviewCamera.gameObject.SetActive(true);
        if (overviewCamera != null) workerCamera.gameObject.SetActive(false);
    }

    // 추가 정보 요청 (버튼 연결)
    public void RequestMoreInfo()
    {
        Debug.Log("추가 정보 요청: " + currentWorkerID);

        // JS 또는 서버 요청 호출 예시
#if UNITY_WEBGL && !UNITY_EDITOR
        Application.ExternalCall("RequestWorkerDetail", currentWorkerID);
#else
        // 또는 유니티 내부에서 서버 호출하는 방식
        // StartCoroutine(RequestInfoFromServer(currentWorkerID));
#endif
    }
}