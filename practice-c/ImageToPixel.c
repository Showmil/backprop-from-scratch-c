#include <stdio.h>  // 입출력 헤더 파일
#include <stdlib.h> // 유틸리티 함수 헤더 파일

#define width 100  // 사진 너비 100
#define height 100 // 사진 높이 100

unsigned char image[54];                                           // 출력할 이미지 담을 배열
unsigned char R[width][height], G[width][height], B[width][height]; // 입력 받은 이미지의 RGB 픽셀 값 저장

int main()
{
    FILE *in = fopen("testImage.png", "rb");    // 테스트 이미지 입력
    FILE *out = fopen("outputImage.png", "wb"); // 출력 이미지 생성

    for (int i = 0; i < 54; i++)
    {
        image[i] = getc(in);
    }

    for (int i = 0; i < width; i++) // 사진의 픽셀값을 (0,0)부터 (54,54까지 탐색)
    {
        for (int j = 0; j < height; j++)
        {
            B[i][j] = getc(in); // RGP 순이 아닌 이유는 윈도우에서 BGR 순으로 저장하기 때문
            G[i][j] = getc(in);
            R[i][j] = getc(in);
        }
    }

    for (int i = 0; i < 54; i++)
    {
        fputc(image[i], out);
    }

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            fputc(B[i][j], out);
            fputc(G[i][j], out);
            fputc(R[i][j], out);
        }
    }

    fclose(in);
    fclose(out);

    return 0;
}