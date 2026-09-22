# Real-time ASCII Camera

Webcam em tempo real convertida para arte ASCII, com rastreamento de mãos via MediaPipe que deforma os caracteres e desenha uma "teia digital" pulsante entre as duas mãos. Um experimento de processamento de vídeo frame a frame e visão computacional em Python.

## Sobre o projeto

O script captura o vídeo da webcam, converte cada frame em uma grade de caracteres ASCII com base no brilho de blocos de pixels e usa o rastreamento de mãos (MediaPipe Hand Landmarker) para:

- Empurrar/deformar os caracteres próximos às mãos, criando um efeito de campo de força ao redor delas.
- Detectar se cada mão está aberta ou fechada (heurística de "punho" baseada na curvatura dos dedos).
- Desenhar uma "teia" de linhas pulsantes conectando as pontas dos dedos correspondentes quando **ambas as mãos estão abertas e visíveis** ao mesmo tempo.

É possível alternar em tempo real entre o modo ASCII e o feed de câmera "cru" (com os efeitos de mão sobrepostos em ambos os casos).

## Como funciona (pipeline)

1. **Captura**: um frame é lido da webcam (`cv2.VideoCapture`) e espelhado horizontalmente.
2. **Detecção de mãos**: o frame (em RGB) é passado ao `HandLandmarker` do MediaPipe, que retorna até 2 mãos com 21 landmarks cada.
3. **Classificação aberta/fechada**: para cada mão, compara-se a distância de 4 dedos (indicador, médio, anelar, mínimo) até o pulso, contra a distância das respectivas juntas PIP — 3 ou mais dedos "curvados" conta como punho fechado.
4. **Conversão para ASCII**: o frame em escala de cinza é dividido em blocos (10×10 px por padrão); o brilho médio de cada bloco escolhe um caractere de uma escala de 11 símbolos (` .,:;=+*#%@`).
5. **Deslocamento pelas mãos**: apenas landmarks de mãos *abertas* aplicam um "empurrão" radial nos caracteres próximos (campo de força com queda suave dentro de um raio configurável), somado a uma leve ondulação senoidal.
6. **Renderização**: os caracteres deslocados são desenhados em um canvas preto (ou sobre o próprio frame da câmera, no modo não-ASCII).
7. **Teia digital**: se as duas mãos estiverem abertas, linhas coloridas (com pulso de cor e "nós de energia") conectam os dedos correspondentes de uma mão à outra.
8. **Exibição**: o canvas final é mostrado em uma janela OpenCV, junto com o FPS atual e o status das mãos detectadas.

## Tecnologias

- **Python 3**
- **OpenCV** (`opencv-python`) — captura de vídeo, processamento de imagem e renderização
- **MediaPipe** (`mediapipe`) — modelo `HandLandmarker` para detecção de landmarks das mãos
- **NumPy** (`numpy`) — operações vetorizadas sobre a grade de caracteres e cálculo dos deslocamentos

As versões usadas estão listadas em [`requirements.txt`](requirements.txt) (sem pin de versão específica).

## Pré-requisitos

- Python 3.7 ou superior
- Uma webcam funcional conectada ao computador
- Conexão com a internet no primeiro uso (para baixar automaticamente o modelo `hand_landmarker.task`, caso ainda não exista na pasta do projeto)

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/akira113p/-Real-time-ASCII-camera.git
cd -Real-time-ASCII-camera
pip install -r requirements.txt
```

## Como executar

```bash
python main.py
```

Na primeira execução, se o arquivo `hand_landmarker.task` não estiver presente, ele será baixado automaticamente dos servidores do MediaPipe.

## Controles

| Tecla | Ação |
|-------|------|
| `q` | Encerra a aplicação |
| `w` | Alterna entre o modo ASCII e o feed de câmera normal (os efeitos de mão continuam ativos em ambos) |

Mostre as duas mãos abertas para a câmera para ativar o efeito de "teia digital" entre os dedos.

## Parâmetros configuráveis

Os principais parâmetros ficam no topo de `main.py` e podem ser ajustados diretamente no código:

- `BLOCK_SIZE`: tamanho (em pixels) de cada bloco convertido em um caractere ASCII.
- `ASCII_CHARS`: conjunto de caracteres usado na escala de brilho.
- `CAMERA_WIDTH` / `CAMERA_HEIGHT`: resolução solicitada à webcam.
- `HAND_INFLUENCE_RADIUS`, `HAND_DISPLACEMENT_STRENGTH`, `HAND_DAMPING`, `MAX_OFFSET_PX`: controlam o alcance e a intensidade do efeito de deslocamento causado pelas mãos.
- `WAVE_AMPLITUDE`, `WAVE_SPATIAL_FREQ`, `WAVE_TIME_FREQ`: parâmetros da ondulação orgânica aplicada sobre a área de influência das mãos.
- `WEB_BASE_COLOR`, `WEB_PULSE_COLOR`, `WEB_THICKNESS`, `WEB_PULSE_SPEED`: aparência e velocidade do pulso das linhas da "teia digital".
- `FIST_CURL_THRESHOLD`: número mínimo de dedos curvados para considerar a mão fechada.

## Observações

- O arquivo `face_landmarker.task` está presente no repositório, mas não é utilizado pelo script atual (`main.py` trabalha apenas com rastreamento de mãos).
- Este é um projeto experimental, criado para explorar processamento de vídeo frame a frame, manipulação vetorizada de imagens com NumPy e detecção de gestos com MediaPipe.

## Licença

O repositório não possui um arquivo de licença explícito no momento.
