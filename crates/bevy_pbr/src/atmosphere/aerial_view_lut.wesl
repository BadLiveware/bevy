import package::{
    atmosphere::{
        bindings::settings,
        functions::{
            sample_density_lut, sample_local_inscattering, uv_to_ray_direction, get_view_position,
            MIDPOINT_RATIO, MIN_EXTINCTION, ABSORPTION_DENSITY, SCATTERING_DENSITY,
        },
    }
};


@group(0) @binding(13) var aerial_view_lut_out: texture_storage_3d<rgba16float, write>;

// Slices a workgroup integrates side by side. A workgroup is one column of
// the LUT (one ray), one invocation a slice: each invocation integrates its
// own slice's segment of the ray, and the segments are joined front to back
// by a scan in workgroup memory. Marching the whole ray in one invocation
// left the 32x32 columns to 1024 invocations walking 320 samples each, in
// four workgroups: most of the GPU idle while they ran.
const SLICES_AT_ONCE: u32 = 32u;

// A segment of the ray: the light it lets through, and the light scattered
// toward the camera within it (as seen from the segment's start).
var<workgroup> through: array<vec3<f32>, SLICES_AT_ONCE>;
var<workgroup> scattered: array<vec3<f32>, SLICES_AT_ONCE>;

@compute
@workgroup_size(1, 1, 32)
fn main(
    @builtin(workgroup_id) column: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let size = settings.aerial_view_lut_size;
    let uv = (vec2<f32>(column.xy) + 0.5) / vec2<f32>(size.xy);
    let ray_dir = uv_to_ray_direction(uv);
    let world_pos = get_view_position();
    let t_max = settings.aerial_view_lut_max_distance;
    let samples = settings.aerial_view_lut_samples;

    // What the slices before this group of them let through and scattered.
    var carry_through = vec3(1.0);
    var carry_scattered = vec3(0.0);

    // (A loop whose bounds are the same for the whole workgroup, so every
    // invocation reaches each barrier.)
    for (var first: u32 = 0u; first < size.z; first += SLICES_AT_ONCE) {
        let slice_i = first + lane;

        var segment_through = vec3(1.0);
        var segment_scattered = vec3(0.0);
        if slice_i < size.z {
            // Where the sample before this slice's first one lies: the steps
            // are taken as one march over the whole ray would take them.
            var prev_t = 0.0;
            if slice_i > 0u {
                prev_t = t_max * (f32(slice_i - 1u) + ((f32(samples - 1u) + MIDPOINT_RATIO) / f32(samples))) / f32(size.z);
            }
            for (var step_i: u32 = 0; step_i < samples; step_i++) {
                let t_i = t_max * (f32(slice_i) + ((f32(step_i) + MIDPOINT_RATIO) / f32(samples))) / f32(size.z);
                let dt = (t_i - prev_t);
                prev_t = t_i;

                let sample_pos = world_pos + ray_dir * t_i;
                let local_r = length(sample_pos);

                let absorption = sample_density_lut(local_r, ABSORPTION_DENSITY);
                let scattering = sample_density_lut(local_r, SCATTERING_DENSITY);
                let extinction = absorption + scattering;

                let sample_optical_depth = extinction * dt;
                let sample_transmittance = exp(-sample_optical_depth);

                // evaluate one segment of the integral
                let inscattering = sample_local_inscattering(scattering, ray_dir, sample_pos);

                // Analytical integration of the single scattering term in the radiance transfer equation
                let s_int = (inscattering - inscattering * sample_transmittance) / max(extinction, MIN_EXTINCTION);
                segment_scattered += segment_through * s_int;
                segment_through *= sample_transmittance;
            }
        }

        // Join the segments front to back (an inclusive scan): a segment
        // behind another is seen through it.
        through[lane] = segment_through;
        scattered[lane] = segment_scattered;
        workgroupBarrier();
        for (var offset: u32 = 1u; offset < SLICES_AT_ONCE; offset *= 2u) {
            var t = through[lane];
            var s = scattered[lane];
            if lane >= offset {
                let front_t = through[lane - offset];
                let front_s = scattered[lane - offset];
                s = front_s + front_t * s;
                t = front_t * t;
            }
            workgroupBarrier();
            through[lane] = t;
            scattered[lane] = s;
            workgroupBarrier();
        }

        if slice_i < size.z {
            let total_inscattering = carry_scattered + carry_through * scattered[lane];
            // Store in log space to allow linear interpolation of exponential values between slices
            let log_inscattering = log(max(total_inscattering, vec3(1e-6)));
            textureStore(aerial_view_lut_out, vec3(column.xy, slice_i), vec4(log_inscattering, 0.0));
        }

        let last = SLICES_AT_ONCE - 1u;
        carry_scattered = carry_scattered + carry_through * scattered[last];
        carry_through = carry_through * through[last];
        workgroupBarrier();
    }
}
