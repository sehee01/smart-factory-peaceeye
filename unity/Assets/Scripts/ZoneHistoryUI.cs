using UnityEngine;
using UnityEngine.UI;

public class ZoneHistoryUI : MonoBehaviour
{
    public static ZoneHistoryUI Instance;

    public GameObject entryPrefab;
    public Transform contentArea;

    void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    public void Display(ZoneHistoryData[] history)
    {
        // 기존 항목 제거
        foreach (Transform child in contentArea)
            Destroy(child.gameObject);

        foreach (var item in history)
        {
            GameObject entry = Instantiate(entryPrefab, contentArea);
            entry.GetComponentInChildren<Text>().text =
                $"시간: {item.hour}시\n" +
                $"평균 사이클: {item.avg_cycle_time:F1}분\n" +
                $"PPE 위반: {item.total_ppe_violations}회\n" +
                $"위험존 체류: {item.total_hazard_dwell}회";
        }
    }
}
