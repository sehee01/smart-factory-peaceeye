using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class TrapezoidFloor : MonoBehaviour
{
    void Start()
    {
        Mesh mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;

        // 사다리꼴 바닥의 꼭짓점 4개 (2D 사다리꼴 기준, Y는 0으로 평면)
        Vector3[] vertices = new Vector3[]
        {
            new Vector3(-1, 0, -1), // 좌하
            new Vector3(1, 0, -1),  // 우하
            new Vector3(-0.5f, 0, 1), // 좌상
            new Vector3(0.5f, 0, 1)   // 우상
        };

        int[] triangles = new int[]
        {
            0, 2, 1,  // 첫 삼각형
            1, 2, 3   // 두 번째 삼각형
        };

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
    }
}
