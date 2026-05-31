import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices found.")
else:
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"\nDevice: {name}  Serial: {serial}")

        for sensor in dev.query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"\n  Sensor: {sensor_name}")
            for profile in sensor.get_stream_profiles():
                vp = profile.as_video_stream_profile()
                print(f"    {profile.stream_name()} | {vp.format()} | {vp.width()}x{vp.height()} @ {vp.fps()}fps")