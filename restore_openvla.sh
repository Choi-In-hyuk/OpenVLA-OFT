#!/bin/bash
echo "Retoring"
rsync -av --delete ~/choi_ws/OpenVLA-OFT/Backup_setting/openvla-oft/ ~/choi_ws/openvla-oft/
echo "Restoring complete!"