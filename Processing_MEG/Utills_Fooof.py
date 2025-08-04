import matplotlib.pyplot as plt
import mne
import numpy as np
from fooof import FOOOFGroup
import scipy as sci
from scipy.interpolate import interp1d
from mne.preprocessing import ICA

avgfunc = np.mean



def pre_process_meg(data, noise):
    raw = data.copy().resample(sfreq=500, n_jobs='cuda', verbose=False).apply_function(sci.signal.detrend,
                                                                                       type='linear')
    if 'VEOG' and 'HEOG' in data.info['ch_names']:
        raw.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog', 'ECG': 'ecg'})
    if 'HEOG' in data.info['ch_names']:
        raw.set_channel_types({'HEOG': 'eog', 'ECG': 'ecg'})
    if 'VEOG' in data.info['ch_names']:
        raw.set_channel_types({'VEOG': 'eog', 'ECG': 'ecg'})
        
    filtered = raw.copy().filter(l_freq=1, h_freq=100, n_jobs='cuda', verbose=False)
    empty_room_projs = mne.compute_proj_raw(noise, n_grad=0.9, n_mag=0.9, n_jobs=8, verbose=False)
    projected_signal = filtered.copy().add_proj(empty_room_projs).apply_proj()
    ''' Removing bad channels and artifacts using ICA
     Automatic ICA labeling (mne.label_components) does not work with MEG data
     In here, we first performed ICA on projected_signal with excluding ref_meg and miscs
     Second, we performed ICA on raw signal with excluding meg, eeg channels'''
    '''https://mne.tools/dev/auto_examples/preprocessing/find_ref_artifacts.html#sphx-glr-auto-examples-preprocessing-find-ref-artifacts-py'''
    # ICA only on MEG channels
    ica_kwargs = dict(
        method="infomax")  # use a high tol here for speed when using picard for methods ==> fit_params=dict(tol=1e-6) faster with lower tol
    ica = ICA(n_components=60, max_iter="auto", allow_ref_meg=False, **ica_kwargs)
    ica.fit(projected_signal.copy().pick_types(meg=True, eeg=True, ref_meg=False, misc=False), verbose=False)
    # a higher threshold (2.5)
    muscle_idx_auto, _ = ica.find_bads_muscle(projected_signal, threshold=1, verbose=False)
    ecg_indices, ecg_scores = ica.find_bads_ecg(projected_signal, verbose=False)
    eog_indices, eog_scores = ica.find_bads_eog(projected_signal, verbose=False)

    # Do ICA only on the reference channels.
    ref_picks = mne.pick_types(raw.info, meg=False, eeg=False, ref_meg=True, misc=False)
    ica_ref = ICA(n_components=len(ref_picks - 1), max_iter="auto", allow_ref_meg=True, **ica_kwargs)
    ica_ref.fit(raw, picks=ref_picks, verbose=False)
    ref_comps = ica_ref.get_sources(raw)
    for c in ref_comps.ch_names:  # they need to have REF_ prefix to be recognised
        ref_comps.rename_channels({c: "REF_" + c})
    raw.add_channels([ref_comps])
    bad_ref, scores = ica.find_bads_ref(raw, method="separate", verbose=False)
    reject_comps = muscle_idx_auto + ecg_indices + eog_indices + bad_ref
    # Remove the components.
    pre_processed = ica.apply(projected_signal, exclude=reject_comps, verbose=False)
    # removing EEG, ECG, EOG, misc signals
    # pre_processed.pick_types(meg=grad, eeg=False, ref_meg=False, misc=False, verbose=False)
    return pre_processed, len(reject_comps)


def cal_foof(data, f_range, low_freq_to_del, high_freq_to_del, remove_line_noise):
    channels = data.ch_names
    srate = int(data.info['sfreq'])
    n_fft, noverlap, n_per_seg = int(20 * srate), int(0.5 * srate), int(10 * srate)
    fg = FOOOFGroup(peak_width_limits=[0, 12], max_n_peaks=7,
                    min_peak_height=0, peak_threshold=0.5, aperiodic_mode='fixed', verbose=False)

    spectrum = data.compute_psd(method='welch', n_fft=n_fft, n_overlap=noverlap, n_per_seg=n_per_seg,
                                fmin=f_range[0], fmax=f_range[1], picks=None,
                                exclude=(), proj=False,
                                remove_dc=True, n_jobs=-1, verbose=False)
    if remove_line_noise:
        psds_noisy, freqs = spectrum.get_data(return_freqs=True)
        idx_low_freq = np.array(np.where(freqs == low_freq_to_del))[0][0]
        idx_high_freq = np.array(np.where(freqs == high_freq_to_del))[0][0]
        psds_noisy[:, :, idx_low_freq:idx_high_freq] = np.nan
        psds = np.zeros((psds_noisy.shape[0], psds_noisy.shape[1], psds_noisy.shape[2]))
        for ep in range(psds_noisy.shape[0]):
            for ch in range(psds_noisy.shape[1]):
                signal = psds_noisy[ep, ch, :]
                f = interp1d(freqs[~np.isnan(signal)], signal[~np.isnan(signal)],
                             fill_value="extrapolate")
                psds[ep, ch, :] = f(freqs)
    else:
        psds, freqs = spectrum.get_data(return_freqs=True)
    '''Averaging over Desirable channels'''

    ''' Separating Regions'''
    # Get MEG channel indices and 3D positions
    meg_channel_indices = mne.pick_types(data.info, meg=True)  # Pick only MEG channels
    channel_positions = np.array([data.info['chs'][i]['loc'][:3] for i in meg_channel_indices])

    # Define regions based on approximate coordinates
    frontal_indices = [i for i, pos in zip(meg_channel_indices, channel_positions) if pos[1] > 0.05]
    occipital_indices = [i for i, pos in zip(meg_channel_indices, channel_positions) if
                         pos[1] < -0.05]  # Occipital: y < -0.05
    central_indices = [i for i, pos in zip(meg_channel_indices, channel_positions) if
                       -0.05 <= pos[1] <= 0.05]  # Central: around y = 0
    left_indices = [i for i, pos in zip(meg_channel_indices, channel_positions) if pos[0] < 0]  # Left Hemisphere: x < 0
    right_indices = [i for i, pos in zip(meg_channel_indices, channel_positions) if
                     pos[0] > 0]  # Right Hemisphere: x > 0

    # Frontal
    frontal_avg = []
    for idx, ch in enumerate(frontal_indices):
        sig = avgfunc(psds[:, ch, :], axis=0)
        frontal_avg.append(sig)
    frontal_psds = avgfunc(np.vstack(frontal_avg), axis=0)
    fg.fit(freqs, np.reshape(frontal_psds, (1, len(frontal_psds))), f_range)
    frontal_foof = fg.get_fooof(0, True)
    

    # Central
    central_avg = []
    for idx, ch in enumerate(central_indices):
        sig = avgfunc(psds[:, ch, :], axis=0)
        central_avg.append(sig)
    central_psds = avgfunc(np.vstack(central_avg), axis=0)
    fg.fit(freqs, np.reshape(central_psds, (1, len(central_psds))), f_range)
    central_foof = fg.get_fooof(0, True)
    

    # Occipital
    occipital_avg = []
    for idx, ch in enumerate(occipital_indices):
        sig = avgfunc(psds[:, ch, :], axis=0)
        occipital_avg.append(sig)
    occipital_psds = avgfunc(np.vstack(occipital_avg), axis=0)
    fg.fit(freqs, np.reshape(occipital_psds, (1, len(occipital_psds))), f_range)
    occipital_foof = fg.get_fooof(0, True)
    
    # Left Hemisphere
    left_avg = []
    for idx, ch in enumerate(left_indices):
        sig = avgfunc(psds[:, ch, :], axis=0)
        left_avg.append(sig)
    left_psds = avgfunc(np.vstack(left_avg), axis=0)
    fg.fit(freqs, np.reshape(left_psds, (1, len(left_psds))), f_range)
    left_foof = fg.get_fooof(0, True)
   

    # Right Hemisphere
    right_avg = []
    for idx, ch in enumerate(right_indices):
        sig = avgfunc(psds[:, ch, :], axis=0)
        right_avg.append(sig)
    right_psds = avgfunc(np.vstack(right_avg), axis=0)
    fg.fit(freqs, np.reshape(right_psds, (1, len(right_psds))), f_range)
    right_foof = fg.get_fooof(0, True)
  

    # All scalp
    all_index = channels
    all_psds = avgfunc(avgfunc(psds, axis=0), axis=0)
    fg.fit(freqs, np.reshape(all_psds, (1, len(all_psds))), f_range)
    all_foof = fg.get_fooof(0, True)
   

    ''' Saving Foof results to dictionary'''
    foof_dict = dict()
    regions = ['Frontal', 'Central', 'Occipital', 'Right', 'Left', 'AllScalp']
    for region in regions:
        foof_dict[region] = dict()
    foof_dict['Frontal'] = frontal_foof
    foof_dict['Central'] = central_foof
    foof_dict['Occipital'] = occipital_foof
    foof_dict['Right'] = right_foof
    foof_dict['Left'] = left_foof
    foof_dict['AllScalp'] = all_foof
    return foof_dict


def signal_plot(data1, data2, save_path, condition):
    regions = ['Frontal', 'Central', 'Occipital', 'Right', 'Left', 'AllScalp']
    groups = ['parkinson', 'healthy']
    # params = ['real_spec', 'foof_spec', 'curve']
    signals = dict()
    for group in groups:
        signals[group] = dict()
        for region in regions:
            signals[group][region] = dict()
            # for param in params:
            #     signals[group][region][param] = dict()
    freqs = data1[f'subj_0']['Frontal'].freqs
    aperiodic_mode = data1[f'subj_0']['Frontal'].aperiodic_mode
    for region in regions:
        spec_parkinson = []
        foof_parkinson = []
        dterend_parkinson = []
        dterend_healthy = []
        curve_parkinson = []
        spec_healthy = []
        foof_healthy = []
        curve_healthy = []
        for subj in range(len(data1)):
            real_spec = data1[f'subj_{subj}'][f'{region}'].power_spectrum
            foof = data1[f'subj_{subj}'][f'{region}'].fooofed_spectrum_
            if aperiodic_mode == 'fixed':
                curve = data1[f'subj_{subj}'][f'{region}'].aperiodic_params_[0] \
                        - np.log10(0.00 + freqs ** data1[f'subj_{subj}'][f'{region}'].aperiodic_params_[1])
            else:
                curve = data1[f'subj_{subj}'][f'{region}'].aperiodic_params_[0] \
                        - np.log10(0.00 + freqs ** data1[f'subj_{subj}'][f'{region}'].aperiodic_params_[2])
            spec_parkinson.append(real_spec)
            foof_parkinson.append(foof)
            curve_parkinson.append(curve)
            dterend_parkinson.append(foof - curve)
        # signals['parkinson'][f'{region}']['real_spec'] = np.array(spec_parkinson)
        signals['parkinson'][f'{region}'] = np.array(dterend_parkinson)
        # signals['parkinson'][f'{region}']['curve'] = np.array(curve_parkinson)

        for subj in range(len(data2)):
            real_spec_sec = data2[f'subj_{subj}'][f'{region}'].power_spectrum
            foof_sec = data2[f'subj_{subj}'][f'{region}'].fooofed_spectrum_
            if aperiodic_mode == 'fixed':
                curve_sec = data2[f'subj_{subj}'][f'{region}'].aperiodic_params_[0] \
                            - np.log10(0.00 + freqs ** data2[f'subj_{subj}'][f'{region}'].aperiodic_params_[1])
            else:
                curve_sec = data2[f'subj_{subj}'][f'{region}'].aperiodic_params_[0] \
                            - np.log10(0.00 + freqs ** data2[f'subj_{subj}'][f'{region}'].aperiodic_params_[2])
            spec_healthy.append(real_spec_sec)
            foof_healthy.append(foof_sec)
            curve_healthy.append(curve_sec)
            dterend_healthy.append(foof_sec - curve_sec)
        signals['healthy'][f'{region}'] = np.array(dterend_healthy)
        # signals['healthy'][f'{region}']['real_spec'] = np.array(spec_healthy)
        # signals['healthy'][f'{region}']['curve'] = np.array(curve_healthy)
        fig, ax1 = plt.subplots(1, 1)
        plt.subplots_adjust(left=0.15, bottom=0.14, right=None, top=None, wspace=None, hspace=None)
        fig1, = ax1.plot(freqs, avgfunc(spec_parkinson, axis=0), color='green')
        fig2, = ax1.plot(freqs, avgfunc(foof_parkinson, axis=0), '--', color='green')
        fig3, = ax1.plot(freqs, avgfunc(curve_parkinson, axis=0), '-.', color='green')
        fig4, = ax1.plot(freqs, avgfunc(spec_healthy, axis=0), color='blue')
        fig5, = ax1.plot(freqs, avgfunc(foof_healthy, axis=0), '--', color='blue')
        fig6, = ax1.plot(freqs, avgfunc(curve_healthy, axis=0), '-.', color='blue')
        ax1.legend([fig1, fig4], [condition[1], condition[2], 'Aperiodic Signal', 'Fitted Curve'])
        ax1.set_title(condition[0] + f' over {region}')
        ax1.set_xlabel('Frequency Hz')
        ax1.set_ylabel('Log amplitude')
        plt.savefig(save_path + f'/FOOOF over {region}.png')

        # Removing Slope
        ''' To convert from log space, 10**x '''
        fig, ax2 = plt.subplots(1, 1)
        fig1, = ax2.plot(np.log10(freqs), (avgfunc(foof_parkinson, axis=0) - avgfunc(curve_parkinson, axis=0)),
                         color='orange')
        fig2, = ax2.plot(np.log10(freqs), (avgfunc(foof_healthy, axis=0) - avgfunc(curve_healthy, axis=0)),
                         color='blue')
        ax2.legend([fig1, fig2], [condition[1], condition[2]])
        ax2.set_title(f'Slope Removed over {region}')
        ax2.set_xlabel('Frequency Hz (log)')
        ax2.set_ylabel('Power')
        plt.savefig(save_path + f'/Slope removed over {region}.png')

        plt.figure()
        plt.grid(True)
        plt.plot(freqs, np.median(dterend_parkinson, axis=0), label='Parkinson', color='green', alpha=0.9)
        plt.fill_between(freqs, np.median(dterend_parkinson, axis=0) -
                         (np.std(dterend_parkinson, axis=0) / np.sqrt(len(dterend_parkinson))),
                         np.median(dterend_parkinson, axis=0) +
                         (np.std(dterend_parkinson, axis=0) / np.sqrt(len(dterend_parkinson))), color='green',
                         alpha=0.1)
        plt.plot(freqs, np.median(dterend_healthy, axis=0), label='Healthy', color='b', alpha=0.9)
        plt.fill_between(freqs, np.median(dterend_healthy, axis=0) -
                         (np.std(dterend_healthy, axis=0) / np.sqrt(len(dterend_healthy))),
                         np.median(dterend_healthy, axis=0) +
                         (np.std(dterend_healthy, axis=0) / np.sqrt(len(dterend_healthy))), color='b', alpha=0.1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('log power')
        plt.title(f'Slope Removed over {region}')
        plt.legend()
        # plt.show()
        plt.savefig(save_path + f'/Slope removed over {region}_withVariance.png')

        # plt.figure()
        # task_related_chages_signal = np.median(signals['parkinson']['AllScalp'], axis=0) - \
        #                              np.median(signals['healthy']['AllScalp'], axis=0)
        # plt.plot(freqs, task_related_chages_signal)
        # plt.xlabel('Frequency (Hz)', fontname='Calibri', fontsize=8)
        # plt.ylabel('log power', fontname='Calibri', fontsize=8)
        # plt.title(f'Related changes over all scalp', fontname='Calibri', fontsize=8)
        # plt.savefig(save_path + 'Related changes over all scalp.png')
        # plt.legend()
    return signals, freqs


def get_area(data):
    areas = dict()
    # harmonic = dict()
    exper = ['parkinson', 'healthy']
    regions = ['Frontal', 'Central', 'Occipital', 'Right', 'Left', 'AllScalp']
    bands = ['alpha', 'beta', 'high_beta']
    features = ['peak', 'CF']

    '''
    alpha = 7-14
    low-beta = 14-20
    high-beta = 20-30
    '''
    # features
    for ex in exper:
        areas[ex] = dict()
        for region in regions:
            areas[ex][region] = dict()
            for band in bands:
                areas[ex][region][band] = dict()
                for feature in features:
                    areas[ex][region][band][feature] = dict()
    for region in regions:
        a1 = data[0]['parkinson'][region].get_results()
        a2 = data[0]['healthy'][region].get_results()

        # Parkinson
        for peaks in range(len(a1[1])):
            f = a1[1][peaks][0]  # freq center
            p = a1[1][peaks][1]  # power
            if 14 > f >= 7:  # Alpha
                areas['parkinson'][region]['alpha']['peak'] = p
                areas['parkinson'][region]['alpha']['CF'] = f
            if 14 <= f < 20:  # low-Beta
                areas['parkinson'][region]['beta']['peak'] = p
                areas['parkinson'][region]['beta']['CF'] = f
            if 20 <= f < 30:  # high-Beta
                areas['parkinson'][region]['high_beta']['peak'] = p
                areas['parkinson'][region]['high_beta']['CF'] = f

        # Healthy
        for peaks in range(len(a2[1])):
            f = a2[1][peaks][0]  # freq center
            p = a2[1][peaks][1]  # power
            if 14 > f >= 7:  # Alpha
                areas['healthy'][region]['alpha']['peak'] = p
                areas['healthy'][region]['alpha']['CF'] = f
            if 14 <= f < 20:  # low-Beta
                areas['healthy'][region]['beta']['peak'] = p
                areas['healthy'][region]['beta']['CF'] = f
            if 20 <= f < 30:  # high-Beta
                areas['healthy'][region]['high_beta']['peak'] = p
                areas['healthy'][region]['high_beta']['CF'] = f

        # harmonic['parkinson'][region] = harmonic_detect(freq_alpha1, freq_beta1)
        # harmonic['healthy'][region] = harmonic_detect(freq_alpha2, freq_beta2)

        # line_graph_xaxis = np.arange(5, 13, 0.25)
        # line_graph_yaxis = line_graph_xaxis * 2

        # fig = plt.figure()
        # plt.grid(True)
        # plt.plot(freq_alpha1, freq_beta1, 'o')
        # plt.plot(line_graph_xaxis, line_graph_yaxis)
        # plt.xlabel(f'Alpha\nalpha/beta harmonic ratio = %{(harmonic_detect(freq_alpha1, freq_beta1)) * 100}')
        # plt.ylabel(f'Beta')
        # plt.title(f'Parkinson alpha/beta harmonic over {region}')
        # plt.subplots_adjust(left=0.15, bottom=0.26, right=None, top=None, wspace=None, hspace=None)
        # plt.savefig(save_root + f'/Parkinson alpha-beta correlation over {region}.png')
        #
        # fig2 = plt.figure()
        # plt.grid(True)
        # plt.plot(freq_alpha2, freq_beta2, 'o')
        # plt.plot(line_graph_xaxis, line_graph_yaxis)
        # plt.xlabel(f'Alpha\nalpha/beta harmonic ratio = %{(harmonic_detect(freq_alpha2, freq_beta2)) * 100}')
        # plt.ylabel(f'Beta')
        # plt.title(f'Healthy alpha/beta harmonic over {region}')
        # plt.subplots_adjust(left=0.15, bottom=0.26, right=None, top=None, wspace=None, hspace=None)
        # plt.savefig(save_root + f'/Healthy alpha-beta correlation over {region}.png')

    return areas


def harmonic_detect(alpha, beta):
    harmon = 0
    for i in range(len(alpha)):
        # print('Len = ', len(alpha), '\n alpha = ', alpha, '\nbeta = ', beta)
        if alpha[i] == beta[i] / 2 or alpha[i] == (beta[i] / 2) + 0.1 or alpha[i] == (beta[i] / 2) - 0.1:
            harmon += 1
    proportion = harmon / len(alpha)
    return proportion


def significant_foof(data1, data2):
    p_values = dict()
    regions = ['Frontal', 'Central', 'Occipital', 'Right', 'Left', 'AllScalp']
    params = ['offset', 'exponent']
    for region in regions:
        p_values[region] = dict()
        for param in params:
            p_values[region][param] = dict()

    p_values['Frontal']['offset'] = sci.stats.mannwhitneyu(
        data1[0, 0, :], data2[0, 0, :]).pvalue
    p_values['Frontal']['exponent'] = sci.stats.mannwhitneyu(
        data1[0, 1, :], data2[0, 1, :]).pvalue
    # Central
    p_values['Central']['offset'] = sci.stats.mannwhitneyu(
        data1[1, 0, :], data2[1, 0, :]).pvalue
    p_values['Central']['exponent'] = sci.stats.mannwhitneyu(
        data1[1, 1, :], data2[1, 1, :]).pvalue
    # Occipital
    p_values['Occipital']['offset'] = sci.stats.mannwhitneyu(
        data1[2, 0, :], data2[2, 0, :]).pvalue
    p_values['Occipital']['exponent'] = sci.stats.mannwhitneyu(
        data1[2, 1, :], data2[2, 1, :]).pvalue
    # Right
    p_values['Right']['offset'] = sci.stats.mannwhitneyu(
        data1[3, 0, :], data2[3, 0, :]).pvalue
    p_values['Right']['exponent'] = sci.stats.mannwhitneyu(
        data1[3, 1, :], data2[3, 1, :]).pvalue
    # Left
    p_values['Left']['offset'] = sci.stats.mannwhitneyu(
        data1[4, 0, :], data2[4, 0, :]).pvalue
    p_values['Left']['exponent'] = sci.stats.mannwhitneyu(
        data1[4, 1, :], data2[4, 1, :]).pvalue
    # AllScalp
    p_values['AllScalp']['offset'] = sci.stats.mannwhitneyu(
        data1[5, 0, :], data2[5, 0, :]).pvalue
    p_values['AllScalp']['exponent'] = sci.stats.mannwhitneyu(
        data1[5, 1, :], data2[5, 1, :]).pvalue
    return p_values
