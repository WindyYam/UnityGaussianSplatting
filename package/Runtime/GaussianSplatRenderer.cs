// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.XR;
using Object = UnityEngine.Object;

namespace GaussianSplatting.Runtime
{
    class GaussianSplatRenderSystem
    {
        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker s_ProfDraw = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCompose = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static GaussianSplatRenderSystem instance => ms_Instance ??= new GaussianSplatRenderSystem();
        static GaussianSplatRenderSystem ms_Instance;

        readonly Dictionary<GaussianSplatRenderer, MaterialPropertyBlock> m_Splats = new();
        readonly HashSet<Camera> m_CameraCommandBuffersDone = new();
        readonly List<(GaussianSplatRenderer, MaterialPropertyBlock)> m_ActiveSplats = new();

        CommandBuffer m_CommandBuffer;
        GraphicsBuffer m_CubeIndexBuffer;
        GraphicsBuffer m_GlobalUniforms;
        Material m_MatSplats;
        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal Material m_MatComposite;
        Material m_MatDebugPoints;
        Material m_MatDebugBoxes;
        uint m_FrameOffset;
        GaussianSplatTemporalFilter m_TemporalFilter;

        struct SplatGlobalUniforms // match cbuffer SplatGlobalUniforms in shaders
        {
            public uint transparencyMode;
            public uint frameOffset;
        }

        public void RegisterSplat(GaussianSplatRenderer r)
        {
            if (m_Splats.Count == 0)
            {
                if (GraphicsSettings.currentRenderPipeline == null)
                    Camera.onPreCull += OnPreCullCamera;
            }

            m_Splats.Add(r, new MaterialPropertyBlock());
        }

        public void UnregisterSplat(GaussianSplatRenderer r)
        {
            if (!m_Splats.ContainsKey(r))
                return;
            m_Splats.Remove(r);
            if (m_Splats.Count == 0)
                CleanupAfterAllSplatsDeleted();
        }

        void CleanupAfterAllSplatsDeleted()
        {
            if (m_CameraCommandBuffersDone != null)
            {
                if (m_CommandBuffer != null)
                {
                    foreach (var cam in m_CameraCommandBuffersDone)
                    {
                        if (cam)
                            cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                    }
                }
                m_CameraCommandBuffersDone.Clear();
            }

            m_ActiveSplats.Clear();
            m_CubeIndexBuffer?.Dispose();
            m_CubeIndexBuffer = null;
            m_CommandBuffer?.Dispose();
            m_CommandBuffer = null;
            m_GlobalUniforms?.Dispose();
            m_GlobalUniforms = null;
            Object.DestroyImmediate(m_MatSplats);
            Object.DestroyImmediate(m_MatComposite);
            Object.DestroyImmediate(m_MatDebugPoints);
            Object.DestroyImmediate(m_MatDebugBoxes);
            m_TemporalFilter?.Dispose();
            m_TemporalFilter = null;
            Camera.onPreCull -= OnPreCullCamera;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            m_ActiveSplats.Clear();
            foreach (var kvp in m_Splats)
            {
                var gs = kvp.Key;
                if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidAsset || !gs.HasValidRenderSetup)
                    continue;
                m_ActiveSplats.Add((kvp.Key, kvp.Value));
            }
            if (m_ActiveSplats.Count == 0)
                return false;

            // sort them by order and depth from camera
            var camTr = cam.transform;
            m_ActiveSplats.Sort((a, b) =>
            {
                var orderA = a.Item1.m_RenderOrder;
                var orderB = b.Item1.m_RenderOrder;
                if (orderA != orderB)
                    return orderB.CompareTo(orderA);
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public void SortAllSplats(Camera cam, CommandBuffer cmb)
        {
            // Sorting is disabled for vertex shader mode with stochastic rendering
            // Only keep sorting for debug box rendering if needed
            if (cam.cameraType == CameraType.Preview)
                return;
            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            if (!settings.needSorting)
                return; // no need to sort

            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;
                var matrix = gs.transform.localToWorldMatrix;
                // Only increment frame counter, no actual sorting for stochastic mode
                ++gs.m_FrameCounter;
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public void CacheViewDataForAllSplats(Camera cam, CommandBuffer cmb)
        {
            // View data calculation is now done in vertex shader, so this method is no longer needed
            // Keep for compatibility with external code that might call it
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public void RenderAllSplats(Camera cam, CommandBuffer cmb)
        {
            EnsureMaterials();
            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            Material displayMat = settings.m_RenderMode switch
            {
                DebugRenderMode.DebugPoints => m_MatDebugPoints,
                DebugRenderMode.DebugPointIndices => m_MatDebugPoints,
                DebugRenderMode.DebugBoxes => m_MatDebugBoxes,
                DebugRenderMode.DebugChunkBounds => m_MatDebugBoxes,
                _ => m_MatSplats
            };
            if (displayMat == null)
                return;

            EnsureCubeIndexBuffer();

            m_GlobalUniforms ??= new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, UnsafeUtility.SizeOf<SplatGlobalUniforms>());
            NativeArray<SplatGlobalUniforms> sgu = new(1, Allocator.Temp);
            sgu[0] = new SplatGlobalUniforms { transparencyMode = (uint)settings.m_Transparency, frameOffset = m_FrameOffset};
            cmb.SetBufferData(m_GlobalUniforms, sgu);
            m_FrameOffset++;

            // Always use stochastic transparency in vertex shader mode
            bool stochastic = !settings.isDebugRender;
            displayMat.SetInt(GaussianSplatRenderer.Props.SrcBlend, (int)(stochastic ? BlendMode.One : BlendMode.OneMinusDstAlpha));
            displayMat.SetInt(GaussianSplatRenderer.Props.DstBlend, (int)(stochastic ? BlendMode.Zero : BlendMode.One));
            displayMat.SetInt(GaussianSplatRenderer.Props.ZWrite, stochastic ? 1 : 0);

            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;

                var matrix = gs.transform.localToWorldMatrix;

                var mpb = kvp.Item2;
                mpb.Clear();

                gs.SetAssetDataOnMaterial(mpb);

                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatChunks, gs.m_GpuChunks);

                // Always use vertex shader mode - pass matrices for vertex shader calculation
                Matrix4x4 matView = cam.worldToCameraMatrix;
                Matrix4x4 matO2W = matrix;
                Matrix4x4 matW2O = matrix.inverse;
                int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
                int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
                Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
                Vector4 camPos = cam.transform.position;
                
                mpb.SetMatrix(GaussianSplatRenderer.Props.MatrixMV, matView * matO2W);
                mpb.SetMatrix(GaussianSplatRenderer.Props.MatrixObjectToWorld, matO2W);
                mpb.SetMatrix(GaussianSplatRenderer.Props.MatrixWorldToObject, matW2O);
                mpb.SetVector(GaussianSplatRenderer.Props.VecScreenParams, screenPar);
                mpb.SetVector(GaussianSplatRenderer.Props.VecWorldSpaceCameraPos, camPos);

                // Set dummy view data buffers for shader compatibility
                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatViewData, gs.m_GpuView);
                mpb.SetBuffer(GaussianSplatRenderer.Props.PrevSplatViewData, gs.m_GpuView);

                mpb.SetBuffer(GaussianSplatRenderer.Props.OrderBuffer, gs.m_GpuSortKeys);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatScale, gs.m_SplatScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatOpacityScale, gs.m_OpacityScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatSize, settings.m_PointDisplaySize);
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOrder, gs.m_SHOrder);
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOnly, settings.m_SHOnly ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayIndex, settings.m_RenderMode == DebugRenderMode.DebugPointIndices ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayChunks, settings.m_RenderMode == DebugRenderMode.DebugChunkBounds ? 1 : 0);
                mpb.SetConstantBuffer(GaussianSplatRenderer.Props.SplatGlobalUniforms, m_GlobalUniforms, 0, m_GlobalUniforms.stride);

                int indexCount = 6;
                int instanceCount = gs.splatCount;
                MeshTopology topology = MeshTopology.Triangles;
                if (settings.m_RenderMode is DebugRenderMode.DebugBoxes or DebugRenderMode.DebugChunkBounds)
                    indexCount = 36;
                if (settings.m_RenderMode == DebugRenderMode.DebugChunkBounds)
                    instanceCount = gs.m_GpuChunksValid ? gs.m_GpuChunks.count : 0;

                cmb.BeginSample(s_ProfDraw);
                cmb.DrawProcedural(m_CubeIndexBuffer, matrix, displayMat, 0, topology, indexCount, instanceCount, mpb);
                cmb.EndSample(s_ProfDraw);
            }
        }

        // cube indices, most often we use only the first quad
        void EnsureCubeIndexBuffer()
        {
            if (m_CubeIndexBuffer != null)
                return;
            m_CubeIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
            m_CubeIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                4, 6, 5, 5, 6, 7,
                0, 2, 4, 4, 2, 6,
                1, 5, 3, 5, 7, 3,
                0, 4, 1, 4, 5, 1,
                2, 3, 6, 3, 7, 6
            });
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal void EnsureMaterials()
        {
            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            if (m_MatSplats == null && settings.resourcesFound)
            {
                m_MatSplats = new Material(settings.shaderSplats) {name = "GaussianSplats"};
                m_MatComposite = new Material(settings.shaderComposite) {name = "GaussianClearDstAlpha"};
                m_MatDebugPoints = new Material(settings.shaderDebugPoints) {name = "GaussianDebugPoints"};
                m_MatDebugBoxes = new Material(settings.shaderDebugBoxes) {name = "GaussianDebugBoxes"};
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        // ReSharper disable once UnusedMethodReturnValue.Global - used by HDRP/URP features that are not always compiled
        public CommandBuffer InitialClearCmdBuffer(Camera cam)
        {
            m_CommandBuffer ??= new CommandBuffer {name = "RenderGaussianSplats"};
            if (GraphicsSettings.currentRenderPipeline == null && cam != null && !m_CameraCommandBuffersDone.Contains(cam))
            {
                cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                m_CameraCommandBuffersDone.Add(cam);
            }

            // get render target for all splats
            m_CommandBuffer.Clear();
            return m_CommandBuffer;
        }

        void OnPreCullCamera(Camera cam)
        {
            if (!GatherSplatsForCamera(cam))
                return;

            EnsureMaterials();
            var matComposite = m_MatComposite;
            if (!matComposite)
                return;

            InitialClearCmdBuffer(cam);

            // We only need this to determine whether we're rendering into backbuffer or not. However, detection this
            // way only works in BiRP so only do it here.
            m_CommandBuffer.SetGlobalTexture(GaussianSplatRenderer.Props.CameraTargetTexture,
                BuiltinRenderTextureType.CameraTarget);

            GaussianSplatSettings settings = GaussianSplatSettings.instance;
            if (!settings.isDebugRender)
            {
                // Set up render targets for temporal filtering - use basic WebGL compatible formats
                m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, -1, -1, 0, FilterMode.Point, RenderTextureFormat.ARGB32);
                m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatMotionRT, -1, -1, 0, FilterMode.Point, RenderTextureFormat.ARGB32);
                m_CommandBuffer.SetRenderTarget(new RenderTargetIdentifier[] { GaussianSplatRenderer.Props.GaussianSplatRT, GaussianSplatRenderer.Props.GaussianSplatMotionRT }, BuiltinRenderTextureType.CurrentActive);
                m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);
            }

            // add sorting, view calc and drawing commands for all splat objects
            SortAllSplats(cam, m_CommandBuffer);
            // View data calculation is now done in vertex shader
            RenderAllSplats(cam, m_CommandBuffer);

            // compose - with temporal filtering if enabled
            if (!settings.isDebugRender)
            {
                m_CommandBuffer.BeginSample(s_ProfCompose);
                if (settings.m_TemporalFilter == TemporalFilter.Temporal)
                {
                    m_TemporalFilter ??= new GaussianSplatTemporalFilter();
                    m_TemporalFilter.Render(m_CommandBuffer, cam, matComposite, 1,
                        GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CameraTarget,
                        settings.m_FrameInfluence, settings.m_VarianceClampScale,
                        GaussianSplatRenderer.Props.GaussianSplatMotionRT);
                }
                else
                {
                    m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
                    m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
                }
                m_CommandBuffer.EndSample(s_ProfCompose);
                m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
                m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatMotionRT);
            }
        }
    }

    [ExecuteInEditMode]
    public class GaussianSplatRenderer : MonoBehaviour
    {
        public GaussianSplatAsset m_Asset;

        [Tooltip("Rendering order compared to other splats. Within same order splats are sorted by distance. Higher order splats render 'on top of' lower order splats.")]
        public int m_RenderOrder;
        [Range(0.1f, 2.0f)] [Tooltip("Additional scaling factor for the splats")]
        public float m_SplatScale = 1.0f;
        [Range(0.05f, 20.0f)]
        [Tooltip("Additional scaling factor for opacity")]
        public float m_OpacityScale = 1.0f;
        [Range(0, 3)] [Tooltip("Spherical Harmonics order to use")]
        public int m_SHOrder = 3;

        int m_SplatCount; // initially same as asset splat count
        internal GraphicsBuffer m_GpuSortKeys; // Keep for shader compatibility but just identity buffer
        GraphicsBuffer m_GpuPosData;
        GraphicsBuffer m_GpuOtherData;
        GraphicsBuffer m_GpuSHData;
        Texture m_GpuColorData;
        internal GraphicsBuffer m_GpuChunks;
        internal bool m_GpuChunksValid;
        internal GraphicsBuffer m_GpuView;
        internal GraphicsBuffer m_GpuViewPrev; // previous frame view data
        internal int m_FrameCounter;
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;
        bool m_Registered;

        internal static class Props
        {
            public static readonly int SrcBlend = Shader.PropertyToID("_SrcBlend");
            public static readonly int DstBlend = Shader.PropertyToID("_DstBlend");
            public static readonly int ZWrite = Shader.PropertyToID("_ZWrite");
            public static readonly int SplatGlobalUniforms = Shader.PropertyToID("SplatGlobalUniforms");
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatOther = Shader.PropertyToID("_SplatOther");
            public static readonly int SplatSH = Shader.PropertyToID("_SplatSH");
            public static readonly int SplatColor = Shader.PropertyToID("_SplatColor");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatViewData = Shader.PropertyToID("_SplatViewData");
            public static readonly int PrevSplatViewData = Shader.PropertyToID("_PrevSplatViewData");
            public static readonly int OrderBuffer = Shader.PropertyToID("_OrderBuffer");
            public static readonly int SplatScale = Shader.PropertyToID("_SplatScale");
            public static readonly int SplatOpacityScale = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int SplatSize = Shader.PropertyToID("_SplatSize");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int SHOrder = Shader.PropertyToID("_SHOrder");
            public static readonly int SHOnly = Shader.PropertyToID("_SHOnly");
            public static readonly int DisplayIndex = Shader.PropertyToID("_DisplayIndex");
            public static readonly int DisplayChunks = Shader.PropertyToID("_DisplayChunks");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int GaussianSplatMotionRT = Shader.PropertyToID("_GaussianSplatMotionRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int VecScreenParams = Shader.PropertyToID("_VecScreenParams");
            public static readonly int VecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
            public static readonly int CameraTargetTexture = Shader.PropertyToID("_CameraTargetTexture");
        }



        public GaussianSplatAsset asset => m_Asset;
        public int splatCount => m_SplatCount;

        enum KernelIndices
        {
            // Keep only essential kernels - editing support removed
        }

        public bool HasValidAsset =>
            m_Asset != null &&
            m_Asset.splatCount > 0 &&
            m_Asset.formatVersion == GaussianSplatAsset.kCurrentVersion &&
            m_Asset.posData != null &&
            m_Asset.otherData != null &&
            m_Asset.shData != null &&
            m_Asset.colorData != null;
        public bool HasValidRenderSetup => m_GpuPosData != null && m_GpuOtherData != null && m_GpuChunks != null;

        const int kGpuViewDataSize = 40;

        void CreateResourcesForAsset()
        {
            if (!HasValidAsset)
                return;

            m_SplatCount = asset.splatCount;
            // For WebGL compatibility, use Vertex target instead of Raw
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, (int) (asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, (int) (asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, (int) (asset.shData.dataSize / 4), 4) { name = "GaussianSHData" };
            m_GpuSHData.SetData(asset.shData.GetData<uint>());
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            // For WebGL compatibility, use simpler texture creation flags
            var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.None) { name = "GaussianColorData" };
            tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
            tex.Apply(false, true);
            m_GpuColorData = tex;
            if (asset.chunkData != null && asset.chunkData.dataSize != 0)
            {
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Vertex,
                    (int) (asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
                m_GpuChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, 1,
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunksValid = false;
            }

            // For WebGL compatibility, we still need view buffers for temporal filtering
            // but they won't be populated via compute shader. Use Vertex target instead of Structured.
            m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, m_Asset.splatCount, kGpuViewDataSize) { name = "GaussianViewData" };
            m_GpuViewPrev?.Dispose();
            m_GpuViewPrev = null;
            
            // Create identity sort keys buffer for shader compatibility (no sorting in stochastic mode)
            m_GpuSortKeys = new GraphicsBuffer(GraphicsBuffer.Target.Vertex, m_Asset.splatCount, 4) { name = "GaussianSortKeys" };
            // Initialize with identity order (0, 1, 2, 3, ...)
            var identityKeys = new uint[m_Asset.splatCount];
            for (uint i = 0; i < m_Asset.splatCount; i++)
            {
                identityKeys[i] = i;
            }
            m_GpuSortKeys.SetData(identityKeys);
        }

        bool resourcesAreSetUp => GaussianSplatSettings.instance.resourcesFound;

        public void EnsureSorterAndRegister()
        {
            if (!m_Registered && resourcesAreSetUp)
            {
                GaussianSplatRenderSystem.instance.RegisterSplat(this);
                m_Registered = true;
            }
        }

        public void OnEnable()
        {
            m_FrameCounter = 0;
            if (!resourcesAreSetUp)
                return;

            EnsureSorterAndRegister();
            CreateResourcesForAsset();
        }



        internal void SetAssetDataOnMaterial(MaterialPropertyBlock mat)
        {
            mat.SetBuffer(Props.SplatPos, m_GpuPosData);
            mat.SetBuffer(Props.SplatOther, m_GpuOtherData);
            mat.SetBuffer(Props.SplatSH, m_GpuSHData);
            mat.SetTexture(Props.SplatColor, m_GpuColorData);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            mat.SetInteger(Props.SplatFormat, (int)format);
            mat.SetInteger(Props.SplatCount, m_SplatCount);
            mat.SetInteger(Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
            mat.SetBuffer(Props.PrevSplatViewData, m_GpuViewPrev ?? m_GpuView);
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        void DisposeResourcesForAsset()
        {
            DestroyImmediate(m_GpuColorData);

            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuSHData);
            DisposeBuffer(ref m_GpuChunks);

            DisposeBuffer(ref m_GpuView);
            DisposeBuffer(ref m_GpuViewPrev);
            DisposeBuffer(ref m_GpuSortKeys);

            m_SplatCount = 0;
            m_GpuChunksValid = false;
        }

        public void OnDisable()
        {
            DisposeResourcesForAsset();
            GaussianSplatRenderSystem.instance.UnregisterSplat(this);
            m_Registered = false;
        }

        public void Update()
        {
            var curHash = m_Asset ? m_Asset.dataHash : new Hash128();
            if (m_PrevAsset != m_Asset || m_PrevHash != curHash)
            {
                m_PrevAsset = m_Asset;
                m_PrevHash = curHash;
                if (resourcesAreSetUp)
                {
                    DisposeResourcesForAsset();
                    CreateResourcesForAsset();
                }
                else
                {
                    Debug.LogError($"{nameof(GaussianSplatRenderer)} component is not set up correctly (Resource references are missing), or platform does not support compute shaders");
                }
            }
        }

        public void ActivateCamera(int index)
        {
            Camera mainCam = Camera.main;
            if (!mainCam)
                return;
            if (!m_Asset || m_Asset.cameras == null)
                return;

            var selfTr = transform;
            var camTr = mainCam.transform;
            var prevParent = camTr.parent;
            var cam = m_Asset.cameras[index];
            camTr.parent = selfTr;
            camTr.localPosition = cam.pos;
            camTr.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
            camTr.parent = prevParent;
            camTr.localScale = Vector3.one;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(camTr);
#endif
        }







    }
}