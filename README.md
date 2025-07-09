# OpenVLA-OFT

## 영상 실행할 때
vlc (영상 이름)

## interactive_libero.py position
~/Downloads$ mv interactive_libero.py ~/choi_ws/openvla-oft/experiments/robot/libero

# LIBERO Interactive Mode - Choose one command to run
# LIBERO-Spatial (공간 관련 태스크)
 python interactive_libero.py \
   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
   --task_suite_name libero_spatial \
   --center_crop True

# LIBERO-Object (객체 조작 태스크)
 python interactive_libero.py \
   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
   --task_suite_name libero_object \
   --center_crop True

# LIBERO-Goal (목표 지향 태스크)
 python interactive_libero.py \
   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
   --task_suite_name libero_goal \
   --center_crop True

# LIBERO-10 (복잡한 장기 태스크)
python interactive_libero.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True