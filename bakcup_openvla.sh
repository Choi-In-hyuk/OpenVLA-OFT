#!/bin/bash
echo "Backing up"
rsync -av --delete ~/choi_ws/openvla-oft/ ~/choi_ws/OpenVLA-OFT/Backup_setting/openvla-oft/
echo "Backup complete!"