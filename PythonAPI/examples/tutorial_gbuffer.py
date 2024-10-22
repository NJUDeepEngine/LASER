#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time



def main():
    actor_list = []

    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.
    # Additionally, we will save all of the gbuffer textures for each frame.

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('vehicle'))

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = world.get_map().get_spawn_points()[0]

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # Let's add now a "rgb" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)

        # Register a callback for whenever a new frame is available. This step is
        # currently required to correctly receive the gbuffer textures, as it is 
        # used to determine whether the sensor is active.
        camera.listen(lambda image: image.save_to_disk('_out/FinalColor-%06d.png' % image.frame))

        # Here we will register the callbacks for each gbuffer texture.
        # The function "listen_to_gbuffer" behaves like the regular listen function,
        # but you must first pass it the ID of the desired gbuffer texture.
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneColor, lambda image: image.save_to_disk('_out/GBuffer-SceneColor-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneDepth, lambda image: image.save_to_disk('_out/GBuffer-SceneDepth-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneStencil, lambda image: image.save_to_disk('_out/GBuffer-SceneStencil-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferA, lambda image: image.save_to_disk('_out/GBuffer-A-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferB, lambda image: image.save_to_disk('_out/GBuffer-B-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferC, lambda image: image.save_to_disk('_out/GBuffer-C-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferD, lambda image: image.save_to_disk('_out/GBuffer-D-%06d.png' % image.frame))
        # Note that some gbuffer textures may not be available for a particular scene.
        # For example, the textures E and F are likely unavailable in this example,
        # which will result in them being sent as black images.
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferE, lambda image: image.save_to_disk('_out/GBuffer-E-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferF, lambda image: image.save_to_disk('_out/GBuffer-F-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.Velocity, lambda image: image.save_to_disk('_out/GBuffer-Velocity-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SSAO, lambda image: image.save_to_disk('_out/GBuffer-SSAO-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.CustomDepth, lambda image: image.save_to_disk('_out/GBuffer-CustomDepth-%06d.png' % image.frame))
        camera.listen_to_gbuffer(carla.GBufferTextureID.CustomStencil, lambda image: image.save_to_disk('_out/GBuffer-CustomStencil-%06d.png' % image.frame))

        time.sleep(10)

    finally:

        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
