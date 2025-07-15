using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class APIClient : MonoBehaviour
{
    public static string jwtToken = "";

    public void SetToken(string token)
    {
        jwtToken = token;
        Debug.Log("JWT 토큰 저장됨: " + jwtToken);
    }

    public void StartPolling()
    {
        InvokeRepeating(nameof(GetUpdates), 1f, 2f); // 2초마다 API 호출
    }

    void GetUpdates()
    {
        StartCoroutine(RequestUpdates());
    }

    IEnumerator RequestUpdates()
    {
        UnityWebRequest req = UnityWebRequest.Get("http://your-server.com/api/updates");
        req.SetRequestHeader("Authorization", "Bearer " + jwtToken);

        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            string json = req.downloadHandler.text;
            Debug.Log("서버 응답: " + json);

            // TODO: 받은 데이터로 작업자 위치, 상태 업데이트
        }
        else
        {
            Debug.LogError("API 요청 실패: " + req.error);
        }
    }
}
