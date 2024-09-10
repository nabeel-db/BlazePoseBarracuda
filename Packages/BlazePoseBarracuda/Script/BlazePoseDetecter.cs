using Mediapipe.PoseDetection;
using Mediapipe.PoseLandmark;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Mediapipe.BlazePose
{
    public class BlazePoseDetecter : System.IDisposable
    {
        #region public variables
        public ComputeBuffer outputBuffer;
        public ComputeBuffer worldLandmarkBuffer;
        public int vertexCount => landmarker.vertexCount;
        #endregion

        #region constant number
        const int DETECTION_INPUT_IMAGE_SIZE = 128;
        const int LANDMARK_INPUT_IMAGE_SIZE = 256;
        const int rvfWindowMaxCount = 5;
        #endregion

        #region private variable
        PoseDetecter detecter;
        PoseLandmarker landmarker;
        ComputeShader cs;
        ComputeBuffer letterboxTextureBuffer, poseRegionBuffer, cropedTextureBuffer;
        ComputeBuffer rvfWindowBuffer, rvfWindowWorldBuffer, lastValueScale, lastValueScaleWorld;
        int rvfWindowCount;
        Vector4[] poseLandmarks, poseWorldLandmarks;

        PoseDetection.PoseDetection[] _post2ReadCache;
        int[] _countReadCache = new int[1];
        #endregion

        #region public method
        public BlazePoseDetecter(BlazePoseModel blazePoseModel = BlazePoseModel.full)
        {
            var resource = Resources.Load<BlazePoseResource>("BlazePose");

            cs = resource.cs;
            detecter = new PoseDetecter(resource.detectionResource);
            landmarker = new PoseLandmarker(resource.landmarkResource, (PoseLandmarkModel)blazePoseModel);

            letterboxTextureBuffer = new ComputeBuffer(DETECTION_INPUT_IMAGE_SIZE * DETECTION_INPUT_IMAGE_SIZE * 3, sizeof(float));
            poseRegionBuffer = new ComputeBuffer(1, sizeof(float) * 24);
            cropedTextureBuffer = new ComputeBuffer(LANDMARK_INPUT_IMAGE_SIZE * LANDMARK_INPUT_IMAGE_SIZE * 3, sizeof(float));

            rvfWindowCount = 0;
            rvfWindowBuffer = new ComputeBuffer(rvfWindowMaxCount * landmarker.vertexCount * 4, sizeof(float));
            rvfWindowWorldBuffer = new ComputeBuffer(rvfWindowMaxCount * landmarker.vertexCount * 4, sizeof(float));

            lastValueScale = new ComputeBuffer(landmarker.vertexCount, sizeof(float) * 3);
            lastValueScaleWorld = new ComputeBuffer(landmarker.vertexCount, sizeof(float) * 3);

            outputBuffer = new ComputeBuffer(landmarker.vertexCount + 1, sizeof(float) * 4);
            worldLandmarkBuffer = new ComputeBuffer(landmarker.vertexCount + 1, sizeof(float) * 4);
            poseLandmarks = new Vector4[landmarker.vertexCount + 1];
            poseWorldLandmarks = new Vector4[landmarker.vertexCount + 1];
        }

        public void ProcessImage(
            Texture inputTexture,
            BlazePoseModel blazePoseModel = BlazePoseModel.full,
            float poseThreshold = 0.75f,
            float iouThreshold = 0.3f)
        {
            var scale = new Vector2(
                Mathf.Max((float)inputTexture.height / inputTexture.width, 1),
                Mathf.Max(1, (float)inputTexture.width / inputTexture.height)
            );
            float deltaTime = 1.0f / (4500.0f * Time.unscaledDeltaTime);

            cs.SetInt("_isLinerColorSpace", QualitySettings.activeColorSpace == ColorSpace.Linear ? 1 : 0);
            cs.SetInt("_letterboxWidth", DETECTION_INPUT_IMAGE_SIZE);
            cs.SetVector("_spadScale", scale);
            cs.SetTexture(0, "_letterboxInput", inputTexture);
            cs.SetBuffer(0, "_letterboxTextureBuffer", letterboxTextureBuffer);
            cs.Dispatch(0, DETECTION_INPUT_IMAGE_SIZE / 8, DETECTION_INPUT_IMAGE_SIZE / 8, 1);

            detecter.ProcessImage(letterboxTextureBuffer, poseThreshold, iouThreshold);

            cs.SetFloat("_deltaTime", deltaTime);
            cs.SetBuffer(1, "_poseDetections", detecter.outputBuffer);
            cs.SetBuffer(1, "_poseDetectionCount", detecter.countBuffer);
            cs.SetBuffer(1, "_poseRegions", poseRegionBuffer);
            cs.Dispatch(1, 1, 1, 1);

            cs.SetTexture(2, "_inputTexture", inputTexture);
            cs.SetBuffer(2, "_cropRegion", poseRegionBuffer);
            cs.SetBuffer(2, "_cropedTextureBuffer", cropedTextureBuffer);
            cs.Dispatch(2, LANDMARK_INPUT_IMAGE_SIZE / 8, LANDMARK_INPUT_IMAGE_SIZE / 8, 1);

            landmarker.ProcessImage(cropedTextureBuffer, (PoseLandmarkModel)blazePoseModel);

            cs.SetInt("_isWorldProcess", 0);
            cs.SetInt("_keypointCount", landmarker.vertexCount);
            cs.SetFloat("_postDeltatime", deltaTime);
            cs.SetInt("_rvfWindowCount", rvfWindowCount);
            cs.SetBuffer(3, "_postInput", landmarker.outputBuffer);
            cs.SetBuffer(3, "_postRegion", poseRegionBuffer);
            cs.SetBuffer(3, "_postRvfWindowBuffer", rvfWindowBuffer);
            cs.SetBuffer(3, "_postLastValueScale", lastValueScale);
            cs.SetBuffer(3, "_postOutput", outputBuffer);
            cs.Dispatch(3, 1, 1, 1);

            cs.SetInt("_isWorldProcess", 1);
            cs.SetInt("_keypointCount", landmarker.vertexCount);
            cs.SetFloat("_postDeltatime", deltaTime);
            cs.SetInt("_rvfWindowCount", rvfWindowCount);
            cs.SetBuffer(3, "_postInput", landmarker.outputBuffer);
            cs.SetBuffer(3, "_postRegion", poseRegionBuffer);
            cs.SetBuffer(3, "_postRvfWindowBuffer", rvfWindowWorldBuffer);
            cs.SetBuffer(3, "_postLastValueScaleWorld", lastValueScaleWorld);
            cs.SetBuffer(3, "_postOutput", worldLandmarkBuffer);
            cs.Dispatch(3, 1, 1, 1);

            outputBuffer.GetData(poseLandmarks);
            worldLandmarkBuffer.GetData(poseWorldLandmarks);

            rvfWindowCount = Mathf.Min(rvfWindowCount + 1, rvfWindowMaxCount);
        }

        public PoseDetection.PoseDetection[] GetDetections()
        {
            // Retrieve count from the countBuffer
            detecter.countBuffer.GetData(_countReadCache, 0, 0, 1);
            var count = _countReadCache[0];

            _post2ReadCache = new PoseDetection.PoseDetection[count];
            detecter.outputBuffer.GetData(_post2ReadCache, 0, 0, count);

            return _post2ReadCache;
        }

        public void Dispose()
        {
            letterboxTextureBuffer.Dispose();
            poseRegionBuffer.Dispose();
            cropedTextureBuffer.Dispose();
            rvfWindowBuffer.Dispose();
            rvfWindowWorldBuffer.Dispose();
            lastValueScale.Dispose();
            lastValueScaleWorld.Dispose();
            outputBuffer.Dispose();
            worldLandmarkBuffer.Dispose();
            detecter.Dispose();
            landmarker.Dispose();
        }
    }
    #endregion
}