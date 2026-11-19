//! The place in the world the GPU's positions are taken about.

use bevy_ecs::prelude::*;
use crate::{extract_resource::ExtractResource, RenderApp};
use bevy_math::{Affine3A, DMat4, Mat4, Vec3, Vec3A, Vec4};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};

/// The place in the world that the positions handed to the GPU are taken
/// about.
///
/// The GPU works in 32-bit floats, which are coarse far from the world's
/// origin: at 8 to 16 km they step by about a millimetre, a pixel at a
/// metre's distance. A view's `clip_from_world` then holds numbers of tens
/// of thousands that cancel in the product with a vertex's world position,
/// and since the temporal jitter of TAA or an upscaler changes the
/// projection every frame, the product rounds otherwise every frame: with
/// the camera still, the picture trembles.
///
/// With this resource in the main world (it is extracted), every position
/// and matrix the engine hands the GPU for a view marked
/// [`UsesRenderOrigin`] (the views of 3D cameras and shadow views) is taken
/// about this point instead of the world's origin: the view's own matrices
/// and position, its frustum, the meshes' transforms of this frame and the
/// last, skins, lights and their shadow cascades, decals, light probes and
/// the atmosphere. What a shader calls world space is then the world less
/// this origin, small numbers near the camera wherever it stands, and
/// [`ViewUniform::render_origin`](super::ViewUniform::render_origin) says
/// the origin to shaders that need true world positions (a pattern fixed
/// in the world).
///
/// Everything on the CPU side (transforms, culling, sorting, the render
/// world's extracted data) stays in world space; only what is uploaded
/// changes. Without the resource, or at zero, nothing changes.
///
/// Keep it near the camera and move it rarely: snapped to a coarse grid, so
/// that what is cached on the GPU across frames in this space is seldom
/// invalidated and a value that is a multiple of a power of two stays exact
/// in 32 bits.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Reflect, ExtractResource)]
#[reflect(Resource, Default, Clone)]
#[extract_app(RenderApp)]
pub struct RenderOrigin(pub Vec3);

/// A render-world view whose GPU data is taken about the [`RenderOrigin`].
///
/// The engine puts it on the views of 3D cameras and on shadow views. Views
/// without it (2D cameras, UI) are drawn in world space as before.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct UsesRenderOrigin;

impl RenderOrigin {
    /// The origin a view is drawn about: this, if the view uses it, else
    /// the world's.
    pub fn for_view(origin: Option<&Self>, uses: bool) -> Vec3 {
        match origin {
            Some(origin) if uses => origin.0,
            _ => Vec3::ZERO,
        }
    }

    /// A matrix that takes world positions (`*_from_world`), made to take
    /// positions about `origin`. The product is formed in double precision,
    /// so that the large translation it held cancels exactly.
    pub fn from_world(origin: Vec3, matrix: Mat4) -> Mat4 {
        if origin == Vec3::ZERO {
            return matrix;
        }
        (matrix.as_dmat4() * DMat4::from_translation(origin.as_dvec3())).as_mat4()
    }

    /// A matrix that gives world positions (`world_from_*`), made to give
    /// positions about `origin`.
    pub fn to_world(origin: Vec3, matrix: Mat4) -> Mat4 {
        if origin == Vec3::ZERO {
            return matrix;
        }
        (DMat4::from_translation(-origin.as_dvec3()) * matrix.as_dmat4()).as_mat4()
    }

    /// An affine transform that gives world positions, made to give
    /// positions about `origin`: its translation less the origin, which is
    /// exact for a translation near the origin.
    pub fn to_world_affine(origin: Vec3, mut affine: Affine3A) -> Affine3A {
        affine.translation -= Vec3A::from(origin);
        affine
    }

    /// A half space `normal . p + d > 0` of world positions, made to hold
    /// positions about `origin`.
    pub fn half_space(origin: Vec3, normal_d: Vec4) -> Vec4 {
        if origin == Vec3::ZERO {
            return normal_d;
        }
        let d = normal_d.w as f64 + normal_d.truncate().as_dvec3().dot(origin.as_dvec3());
        normal_d.truncate().extend(d as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    #[test]
    fn a_matrix_about_the_origin_takes_the_same_point_to_the_same_place() {
        let origin = Vec3::new(-3584.0, 0.0, 9984.0);
        let world_from_view = Affine3A::from_rotation_translation(
            bevy_math::Quat::from_rotation_y(0.7),
            Vec3::new(-3571.7, 41.3, 9956.4),
        );
        let view_from_world = Mat4::from(world_from_view.inverse());
        let about = Mat4::from(RenderOrigin::to_world_affine(origin, world_from_view)).inverse();
        let p = Vec3::new(-3572.2, 41.0, 9955.9);
        let near = view_from_world.transform_point3(p);
        let from_about = about.transform_point3(p - origin);
        assert!((near - from_about).length() < 1e-3, "{near} {from_about}");
        let shifted = RenderOrigin::from_world(origin, view_from_world);
        assert!((shifted.transform_point3(p - origin) - near).length() < 1e-3);
    }

    #[test]
    fn a_half_space_about_the_origin_holds_the_same_points() {
        let origin = Vec3::new(1024.0, 0.0, -2048.0);
        let plane = Vec3::new(0.6, 0.0, 0.8).extend(-300.0);
        let p = Vec3::new(1100.0, 5.0, -1600.0);
        let world = plane.truncate().dot(p) + plane.w;
        let shifted = RenderOrigin::half_space(origin, plane);
        let about = shifted.truncate().dot(p - origin) + shifted.w;
        assert!((world - about).abs() < 1e-2, "{world} {about}");
    }
}
