use bevy_camera::{MainPassResolutionOverride, Viewport};
use bevy_core_pipeline::core_3d::{AlphaMask3d, Opaque3d, Transparent3d};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::UVec2;
use bevy_platform::collections::HashMap;
use bevy_render::{
    camera::ExtractedCamera,
    render_phase::{ViewBinnedRenderPhases, ViewSortedRenderPhases},
    render_resource::{
        Extent3d, FilterMode, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
        TextureDimension, TextureUsages, TextureView,
    },
    renderer::RenderDevice,
    texture::TextureCache,
    view::ExtractedView,
};

use crate::{ScreenSpaceTransmission, Transmissive3d};

#[derive(Component)]
pub struct ViewTransmissionTexture {
    pub texture: Texture,
    pub view: TextureView,
    pub sampler: Sampler,
}

pub fn prepare_core_3d_transmission_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    opaque_3d_phases: Res<ViewBinnedRenderPhases<Opaque3d>>,
    alpha_mask_3d_phases: Res<ViewBinnedRenderPhases<AlphaMask3d>>,
    transmissive_3d_phases: Res<ViewSortedRenderPhases<Transmissive3d>>,
    transparent_3d_phases: Res<ViewSortedRenderPhases<Transparent3d>>,
    views_3d: Query<(
        Entity,
        &ExtractedCamera,
        &ScreenSpaceTransmission,
        &ExtractedView,
        Option<&MainPassResolutionOverride>,
    )>,
) {
    // The part of each target the main passes draw into: the copy holds that
    // and no more. Under an upscaler the main pass draws a corner of the
    // target (`MainPassResolutionOverride`), and the rest of a copy of the
    // whole target was never read: at 5120x2160 drawn at 3413x1440, 47 MB
    // of 85. Shaders reach the copy through its own size
    // (`textureDimensions`), so a smaller one reads the same pixels.
    let mut extents = <HashMap<_, UVec2>>::default();
    for (_, camera, transmission, _, resolution_override) in &views_3d {
        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };
        if transmission.steps == 0 {
            continue;
        }
        let extent = Viewport::from_viewport_and_override(
            camera.viewport.as_ref(),
            resolution_override,
        )
        .map_or(physical_target_size, |viewport| {
            (viewport.physical_position + viewport.physical_size).min(physical_target_size)
        })
        .max(UVec2::ONE);
        extents
            .entry(camera.target.clone())
            .and_modify(|e: &mut UVec2| *e = e.max(extent))
            .or_insert(extent);
    }

    let mut textures = <HashMap<_, _>>::default();
    for (entity, camera, transmission, view, _) in &views_3d {
        if !opaque_3d_phases.contains_key(&view.retained_view_entity)
            || !alpha_mask_3d_phases.contains_key(&view.retained_view_entity)
            || !transparent_3d_phases.contains_key(&view.retained_view_entity)
        {
            continue;
        };

        let Some(transmissive_3d_phase) = transmissive_3d_phases.get(&view.retained_view_entity)
        else {
            continue;
        };

        // Don't prepare a transmission texture if the number of steps is set to 0
        if transmission.steps == 0 {
            continue;
        }

        let Some(&extent) = extents.get(&camera.target) else {
            continue;
        };

        // Don't prepare a transmission texture if there are no transmissive items to render
        if transmissive_3d_phase.items.is_empty() {
            continue;
        }

        let cached_texture = textures
            .entry(camera.target.clone())
            .or_insert_with(|| {
                let usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

                let descriptor = TextureDescriptor {
                    label: Some("view_transmission_texture"),
                    // The part of the target the main passes draw into
                    size: Extent3d {
                        width: extent.x,
                        height: extent.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1, // No need for MSAA, as we'll only copy the main texture here
                    dimension: TextureDimension::D2,
                    format: view.target_format,
                    usage,
                    view_formats: &[],
                };

                texture_cache.get(&render_device, descriptor)
            })
            .clone();

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("view_transmission_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        commands.entity(entity).insert(ViewTransmissionTexture {
            texture: cached_texture.texture,
            view: cached_texture.default_view,
            sampler,
        });
    }
}
