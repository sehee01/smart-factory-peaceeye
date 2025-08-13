
import tensorrt as trt
import os

# TensorRT 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, precision_mode='fp16'):
    """
    ONNX 모델을 TensorRT 엔진으로 빌드합니다.

    :param onnx_file_path: 입력 ONNX 파일 경로
    :param engine_file_path: 출력 TensorRT 엔진 파일 경로
    :param precision_mode: 'fp32', 'fp16', 'int8' 중 정밀도 모드 설정
    """
    # 빌더, 네트워크, 파서 생성
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX 모델 파일 로드 및 파싱
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file not found at {onnx_file_path}")
        return None

    print(f"Loading ONNX file from {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file.")

    # 빌더 설정
    config = builder.create_builder_config()
    
    # 최대 작업 공간 크기 설정 (메모리 사용량)
    # 이 값은 모델과 GPU에 따라 조절해야 할 수 있습니다.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31) # 2GB

    # 정밀도 설정
    if precision_mode == 'fp16' and builder.platform_has_fast_fp16:
        print("Using FP16 precision.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision_mode == 'int8' and builder.platform_has_fast_int8:
        print("Using INT8 precision.")
        # INT8 보정(calibration) 과정이 필요합니다. 여기서는 단순화를 위해 생략합니다.
        # 실제 사용 시에는 보정 데이터셋을 이용한 캘리브레이터 구현이 필요합니다.
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        print("Using FP32 precision.")

    print("Building engine...")
    # 엔진 빌드
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the engine.")
        return None
    
    print(f"Completed building engine. Writing to {engine_file_path}")
    # 빌드된 엔진을 파일에 저장
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
        
    print("Engine file saved successfully.")
    return serialized_engine

if __name__ == '__main__':
    # ONNX 파일과 생성될 엔진 파일 경로 설정
    onnx_model_path = os.path.join(os.path.dirname(__file__), 'weights', 'best.onnx')
    engine_model_path = os.path.join(os.path.dirname(__file__), 'weights', 'best.engine')

    # 엔진 빌드 함수 호출
    build_engine(onnx_model_path, engine_model_path, precision_mode='fp16')
