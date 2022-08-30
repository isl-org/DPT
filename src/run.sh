# cmake -DCMAKE_PREFIX_PATH=./libtorch ..
cmake --build ./build --config Release
./build/main \
    /home/tmc/project/DPT_with_onnx_export/DPT/traced_model.pt \
   /home/tmc/Documents/Data/depth_test_data/A011_04010827_C017/000206.jpg \
   /home/tmc/Documents/Data/depth_test_data/A011_04010827_C017