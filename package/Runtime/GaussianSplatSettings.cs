// SPDX-License-Identifier: MIT

using UnityEngine;

namespace GaussianSplatting.Runtime
{
    public enum TransparencyMode
    {
        // no sorting, transparency is stochastic (random) and noisy
        Stochastic,
        // no sorting, transparency is stochastic (half tone) and less noisy
        StochasticHalfTone,
    }

    public enum TemporalFilter
    {
        None,
        Temporal,
    }

    public enum DebugRenderMode
    {
        Splats,
        DebugPoints,
        DebugPointIndices,
        DebugBoxes,
        DebugChunkBounds,
    }

    // If an object with this script exists in the scene, then global 3DGS rendering options
    // are used from that script. Otherwise, defaults are used.
    //
    [ExecuteInEditMode] // so that Awake is called in edit mode
    [DefaultExecutionOrder(-100)]
    public class GaussianSplatSettings : MonoBehaviour
    {
        public static GaussianSplatSettings instance
        {
            get
            {
                if (ms_Instance == null)
                    ms_Instance = FindAnyObjectByType<GaussianSplatSettings>();
                if (ms_Instance == null)
                {
                    var go = new GameObject($"{nameof(GaussianSplatSettings)} (Defaults)")
                    {
                        hideFlags = HideFlags.HideAndDontSave
                    };
                    ms_Instance = go.AddComponent<GaussianSplatSettings>();
                    ms_Instance.EnsureResources();
                }
                return ms_Instance;
            }
        }
        static GaussianSplatSettings ms_Instance;

        [Tooltip("Gaussian splat transparency rendering algorithm")]
        public TransparencyMode m_Transparency = TransparencyMode.Stochastic;

        [Tooltip("How to filter temporal transparency")]
        public TemporalFilter m_TemporalFilter = TemporalFilter.Temporal;
        [Tooltip("How much of new frame to blend in. Higher: more noise, lower: more ghosting.")]
        [Range(0.001f, 1.0f)] public float m_FrameInfluence = 0.05f;
        [Tooltip("Strength of history color rectification clamp. Lower: more flickering, higher: more blur/ghosting.")]
        [Range(0.001f, 10.0f)] public float m_VarianceClampScale = 1.5f;

        public DebugRenderMode m_RenderMode = DebugRenderMode.Splats;
        [Range(1.0f,15.0f)] public float m_PointDisplaySize = 3.0f;
        [Tooltip("Show only Spherical Harmonics contribution, using gray color")]
        public bool m_SHOnly;

        // Remove the vertex shader mode option since it's now the only mode
        // [Tooltip("Use vertex shader mode for better WebGL compatibility (disables compute shaders and temporal filtering)")]
        // public bool m_UseVertexShaderMode;

        internal bool isDebugRender => m_RenderMode != DebugRenderMode.Splats;

        // Sorting is no longer needed since we only use stochastic rendering
        internal bool needSorting => m_RenderMode == DebugRenderMode.DebugBoxes;

        internal bool resourcesFound { get; private set; }
        bool resourcesLoadAttempted;
        internal Shader shaderSplats { get; private set; }
        internal Shader shaderComposite { get; private set; }
        internal Shader shaderDebugPoints { get; private set; }
        internal Shader shaderDebugBoxes { get; private set; }
        // Compute shader is optional now since we use vertex shader mode
        internal ComputeShader csUtilities { get; private set; }

        void Awake()
        {
            if (ms_Instance != null && ms_Instance != this)
                DestroyImmediate(ms_Instance.gameObject);
            ms_Instance = this;
            EnsureResources();
        }

        void EnsureResources()
        {
            if (resourcesLoadAttempted)
                return;
            resourcesLoadAttempted = true;

            shaderSplats = Resources.Load<Shader>("GaussianSplats");
            shaderComposite = Resources.Load<Shader>("GaussianComposite");
            shaderDebugPoints = Resources.Load<Shader>("GaussianDebugRenderPoints");
            shaderDebugBoxes = Resources.Load<Shader>("GaussianDebugRenderBoxes");
            // Do not load compute shader - compute support is intentionally stripped; renderer uses vertex/fragment shaders only
            // csUtilities = Resources.Load<ComputeShader>("GaussianSplatUtilities");

            resourcesFound =
                shaderSplats != null && shaderComposite != null && shaderDebugPoints != null && shaderDebugBoxes != null;
            // Compute shaders are optional for vertex shader mode
            UpdateGlobalOptions();
        }

        void OnValidate()
        {
            UpdateGlobalOptions();
        }

        void OnDidApplyAnimationProperties()
        {
            UpdateGlobalOptions();
        }

        void UpdateGlobalOptions()
        {
            // nothing just yet
        }
    }
}
