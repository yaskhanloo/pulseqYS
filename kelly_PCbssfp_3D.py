# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:50:31 2024

@author: Kelly
"""
import math
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt

fov=[190e-3, 190e-3, 190e-3]     # Define FOV
Nx=64
Ny=Nx
Nz=Nx      # Define FOV and resolution

adc_dur = 2560 #us # ADC duration (controls TR/TE)
alpha = 40  # flip angle
slice_thickness = 3e-3  # slice
n_slices = 1
rf_dur=600 #us
rf_apo=0.5
rf_bwt=1.5
    
Tread=3.2e-3
Tpre=3e-3
riseTime=400e-6
Ndummy=50

# define system properties
system = pp.Opts(
    max_grad=20,
    grad_unit="mT/m",
    #riseTime = "riseTime",
    max_slew=150,
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)
 
seq=pp.Sequence(system) #Create a new sequence object


_, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=alpha * np.pi / 180,
    duration = 3e-3, #rf_dur*1e-6,
    slice_thickness = slice_thickness, #*1e-3,
    apodization=rf_apo,
    time_bw_product=rf_bwt,
    system=system,
    return_gz=True,
    )
# Create non-selective pulse
[rf, ref_delay] = pp.make_block_pulse(flip_angle=8*np.pi/180,system=system,duration=0.2e-3,return_delay=True)


# Define other gradients and ADC events
#delta_k = 1 / fov
delta_k = [1 / val for val in fov]
gx = pp.make_trapezoid(
    channel="x", flat_area=Nx * delta_k[0], flat_time=adc_dur*1e-6, system=system
)
adc = pp.make_adc(
    num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
)
gx_pre = pp.make_trapezoid(
    channel="x", area=-gx.area / 2, duration=1e-3, system=system
)
phase_areas_Y = (np.arange(Ny) - Ny / 2) * delta_k[1]
phase_areas_Z = (np.arange(Ny) - Ny / 2) * delta_k[2]


################################################################3
gz_parts = pp.split_gradient_at(gz, pp.calc_duration(rf))
gz_parts[0].delay = pp.calc_duration(gz_reph)
gz_1 = pp.add_gradients((gz_reph,gz_parts[0]),system=system)
rf, _ = pp.align(right=[rf, gz_1])
gz_parts[1].delay = 0
gz_reph.delay= pp.calc_duration(gz_parts[1])
gz_2 = pp.add_gradients((gz_parts[1],gz_reph),system=system)

gx_parts = pp.split_gradient_at(gx,np.ceil(pp.calc_duration(adc)/system.grad_raster_time)*system.grad_raster_time)
gx_parts[0].delay = pp.calc_duration(gx_pre)
gx_1 = pp.add_gradients((gx_pre,gx_parts[0]),system=system)
adc.delay = adc.delay + pp.calc_duration(gx_pre) # we cannot use mr.align here because the adc duration maz be not aligned to the grad raster
gx_parts[1].delay = 0
gx_pre.delay = pp.calc_duration(gx_parts[1])
gx_2 = pp.add_gradients((gx_parts[1],gx_pre),system=system)
###################################################################

# Calculate timing
gx_pre.delay = 0 # otherwise duration below is misreported
pe_dur = pp.calc_duration(gx_2) # phase encoding duration

# adjust delays to align objects
gz_1.delay = max(pp.calc_duration(gx_2) - rf.delay + rf.ringdown_time,0) # this rf.ringdownTime is needed to center the ADC and the gradient echo in the center of RF-RF period
rf.delay=rf.delay+gz_1.delay

# finish timing calculation
TR=pp.calc_duration(gz_1) + pp.calc_duration(gx_1)
TE=TR/2

# Create 0.5*alpha pulse
rf05=rf
rf05.signal=0.5*rf.signal
seq.add_block(rf05,gz_1,pp.make_label('REV','SET', 1))
gx_2.area = sum(gx_2.waveform)*system.grad_raster_time
gy_pre2 = pp.make_trapezoid(channel='y',area=phase_areas_Y[-1],duration=pe_dur,system=system) #last PE step (in case of repetitions)

# ======
# CONSTRUCT SEQUENCE
# ======
# Loop over phase encodes and define sequence blocks

# Phase Cycles
pcs = np.arange(0, 360, 60)
pcs = -pcs / 180 *(np.pi)
rf_phase = -np.pi
    
# Loops
for pc in pcs:
    for pre in range(1000):   # Dummy Pulses
        rf.phase_offset = np.mod(rf.phase_offset + pc, 2*np.pi)
        adc.phase_offset = np.mod(adc.phase_offset + pc + np.pi/2, 2*np.pi)
        gy_pre1 = pp.make_trapezoid(channel="y",area=0,duration=pe_dur,system=system)
        
        #seq.add_block(rf, gz_1, gy_pre1, gx_2) 
        #seq.add_block(gx_1, gy_pre1, gz_2) 
        
        seq.add_block(rf,ref_delay)
        seq.add_block(gx_1)
        seq.add_block(gx_2)
        
    for s in range(n_slices): #3D
        rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        gz_pre = pp.make_trapezoid(
            channel="z", area=phase_areas_Z[s], system=system
            )    
        gz_reph = pp.make_trapezoid(
            channel="z", area= - phase_areas_Z[s], system=system
            )          
        #Loop over phase encodes
        for i in range(Ny):
            # add another loop for diff TEs
            seq.add_block(rf,ref_delay)
            gy_pre1 = pp.scale_grad(grad=gy_pre2, scale=-1) # Undo previous PE step
            gy_pre2 = pp.make_trapezoid(channel='y', area=phase_areas_Y[i], duration=pe_dur, system=system) # Update gy_pre2 for the current PE step

            #seq.add_block(rf, gz_1, gy_pre2, gx_2) # -->  excitation and encoding phase
            #seq.add_block(gx_1, gy_pre1, gz_2, adc) # -->  readout and data acquisition phase 
            
            seq.add_block(gx_1,gy_pre1,gz_pre,adc)
            seq.add_block(gx_2,gy_pre2, gz_reph)
                
#  finish the x-grad shape 
#seq.add_block(gx_2,pp.make_label('REV','SET', 2)) # we also label this block as the exit block, which excludes it from all but last repetitions if the sequence is repeated

# check that the calculated TR was reached
# alpha / 2 prep takes 4 blocks


# check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed! Error listing follows:")
for error in error_report:
    print(error)

#
TR = (pp.calc_duration(seq.get_block(5)) + pp.calc_duration(seq.get_block(6)))
#assert(TR==(pp.calc_duration(seq.get_block(5))+pp.calc_duration(seq.get_block(6))))
print('Sequence ready')
print(f'TR={TR * 1e3:.3f} ms  TE={TE * 1e3:.3f} ms')
print(gy_pre1.amplitude)
#

# ======
# VISUALIZATION
# ======

#if plot:
#seq.plot()

seq.plot(label="lin", time_range=np.array([1000, 1025]) * TR, time_disp="ms")
#seq.calculate_kspace()

# Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
# slew-rate limits
rep = seq.test_report()
print('test report: ', rep)

'''
# =========
# WRITE .SEQ
# =========
if write_seq:
    # Prepare the sequence output for the scanner
    seq.set_definition(key="FOV", value=[fov, fov, slice_thickness * n_slices])
    seq.set_definition(key="Name", value="gre")

    seq.write(seq_filename)
'''


#%%
"""
## plots and displays
seq.plot(time_range=[1, 1000]*TR)
## plot entire interpolated wave forms -- good for debugging of gaps, jumps,
# etc, but is relatively slow
#gw=seq.gradient_waveforms();
#figure; plot(gw'); % plot the entire gradient waveform
gw_data=seq.waveforms_and_times()
#plt.plot(gw_data{1}(1,:),gw_data{1}(2,:),gw_data{2}(1,:),gw_data{2}(2,:),gw_data{3}(1,:),gw_data{3}(2,:)) # plot the entire gradient waveform
#plt.title('gradient waveforms')
for data in gw_data:
    plt.plot(data[0], data[1])
# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gradient Waveform')
plt.grid(True)
plt.show()

## trajectory calculation 
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculate_kspacePP()


# plot k-spaces
'''
plt.plot(t_ktraj,ktraj) 
plt.title('k-space components as functions of time') # plot the entire k-space trajectory
plt.plot(ktraj(1,:),ktraj(2,:),'b',...
             ktraj_adc(1,:),ktraj_adc(2,:),'r.') # a 2D plot
plt.title('2D k-space')
'''
# Plot k-space components as functions of time
plt.figure(figsize=(10, 5))
plt.plot(t_ktraj, ktraj)
plt.title('k-space components as functions of time')
plt.xlabel('Time')
plt.ylabel('k-space')
plt.grid(True)
plt.show()
# Plot 2D k-space
plt.figure(figsize=(8, 6))
plt.plot(ktraj[0], ktraj[1], 'b', label='ktraj')
plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.', label='ktraj_adc')
plt.title('2D k-space')
plt.xlabel('kx')
plt.ylabel('ky')
plt.legend()
plt.grid(True)
plt.show()

## very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within slewrate limits  
'''
rep = seq.testReport
fprintf([rep{:}])
'''
for report in seq.testReport:
    print(report, end='')
##
'''
phaseAreas = ((0:Ny-1)-Ny/2)*deltak
pp.traj2grad(phaseAreas)
'''
phase_areas_Y = (np.arange(Ny) - Ny / 2) * delta_k[1]
pp.traj2grad(phase_areas_Y)

"""






















 

# %%
