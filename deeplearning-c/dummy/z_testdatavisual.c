#include <stdio.h>
#include <stdlib.h>

#define ANSI_RESET "\033[0m"

// MNIST 파일은 Big-Endian이므로 Little-Endian 시스템(PC)에서 읽기 위해 변환 필요
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int main() {
    // 1. 학습용(Train) 데이터셋 파일 열기
    FILE *imgFile = fopen("train-images.idx3-ubyte", "rb");
    FILE *lblFile = fopen("train-labels.idx1-ubyte", "rb");

    if (!imgFile || !lblFile) {
        printf("파일을 찾을 수 없습니다. 'train-images.idx3-ubyte'와 'train-labels.idx1-ubyte' 파일이 같은 폴더에 있는지 확인하세요.\n");
        return 1;
    }

    // 2. 헤더 정보 읽기
    int magic, numImgs, rows, cols, lblMagic, numLbls;
    
    fread(&magic, 4, 1, imgFile);
    fread(&numImgs, 4, 1, imgFile);
    fread(&rows, 4, 1, imgFile);
    fread(&cols, 4, 1, imgFile);
    
    fread(&lblMagic, 4, 1, lblFile);
    fread(&numLbls, 4, 1, lblFile);

    // 엔디언 변환
    numImgs = reverseInt(numImgs);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    printf("훈련 데이터셋 정보\n");
    printf("- 총 이미지: %d 개\n", numImgs);
    printf("- 이미지 크기: %d x %d\n", cols, rows);
    printf("-------------------------------------------\n");

    // 3. 상위 10개 데이터 출력
    int imgSize = rows * cols;
    unsigned char *pixels = (unsigned char *)malloc(imgSize);
    unsigned char label;

    for (int i = 0; i < 1; i++) {
        fread(pixels, 1, imgSize, imgFile);
        fread(&label, 1, 1, lblFile);

        printf("[Train Data #%d] 정답(Label): %d\n", i + 1, label);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                unsigned char p = pixels[r * cols + c];
                
                // 픽셀 값에 따라 명암 표현 (ANSI 256색상 활용)
                if (p > 200) printf("\033[48;2;255;255;255m  "); // 아주 밝음
                else if (p > 100) printf("\033[48;2;150;150;150m  "); // 중간
                else if (p > 0) printf("\033[48;2;80;80;80m  ");   // 어두움
                else printf(ANSI_RESET "  "); // 배경(검정)
            }
            printf(ANSI_RESET "\n");
        }
        printf("-------------------------------------------\n");
    }

    // 4. 메모리 해제
    free(pixels);
    fclose(imgFile);
    fclose(lblFile);

    printf("출력이 완료되었습니다.\n");
    return 0;
}