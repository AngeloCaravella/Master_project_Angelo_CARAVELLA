@echo off
REM Deletes the old log file to start fresh
if exist log.txt del log.txt

echo Starting image generation... > log.txt

echo --- Generating Interaction Loop diagrams... --- >> log.txt
call mmdc -i ddpg_interaction_loop.mmd -o ddpg_interaction_loop.png >> log.txt 2>&1
call mmdc -i sac_interaction_loop.mmd -o sac_interaction_loop.png >> log.txt 2>&1
call mmdc -i per_interaction_loop.mmd -o per_interaction_loop.png >> log.txt 2>&1

echo --- Generating DDPG diagrams... --- >> log.txt
call mmdc -i ddpg_part1_critic.mmd -o ddpg_part1_critic.png >> log.txt 2>&1
call mmdc -i ddpg_part2_actor.mmd -o ddpg_part2_actor.png >> log.txt 2>&1

echo --- Generating SAC diagrams... --- >> log.txt
call mmdc -i sac_part1_setup.mmd -o sac_part1_setup.png >> log.txt 2>&1
call mmdc -i sac_part2_updates.mmd -o sac_part2_updates.png >> log.txt 2>&1

echo --- Generating PER diagrams... --- >> log.txt
call mmdc -i per_part1_sampling.mmd -o per_part1_sampling.png >> log.txt 2>&1
call mmdc -i per_part2_correction.mmd -o per_part2_correction.png >> log.txt 2>&1

echo --- Script finished. See log.txt for details. --- >> log.txt

echo Script finished. Please check the folder for the 9 images.
pause