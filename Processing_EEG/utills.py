import matplotlib.pyplot as plt
import mne
import numpy as np
from fooof import FOOOFGroup
import scipy as sci
from scipy.interpolate import interp1d

avgfunc = np.mean




def cal_foof(data, f_range, low_freq_to_del, high_freq_to_del, remove_line_noise):
    channels = data.ch_names
    srate = int(data.info['sfreq'])
    n_fft, noverlap, n_per_seg = int(20 * srate), int(0.5 * srate), int(10 * srate)
    fg = FOOOFGroup(peak_width_limits=[0, 12], max_n_peaks=7,
                    min_peak_height=0, peak_threshold=1, aperiodic_mode='fixed', verbose=False)

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

    # Frontal
    frontal_index = np.hstack([channels[0:3], channels[23:28]])
    frontal_psds = avgfunc(np.vstack([avgfunc(psds[:, 0:3, :], axis=0), avgfunc(psds[:, 23:28, :], axis=0)]), axis=0)
    fg.fit(freqs, np.reshape(frontal_psds, (1, len(frontal_psds))), f_range)
    frontal_foof = fg.get_fooof(0, True)

    # Central
    central_index = np.hstack([channels[18:21], channels[5:7]])
    central_psds = avgfunc(np.vstack([avgfunc(psds[:, 18:21, :], axis=0), avgfunc(psds[:, 5:7, :], axis=0)]), axis=0)
    fg.fit(freqs, np.reshape(central_psds, (1, len(central_psds))), f_range)
    central_foof = fg.get_fooof(0, True)

    # Occipital
    occipital_index = np.hstack([channels[9:14], channels[14:16]])
    occipital_psds = avgfunc(np.vstack([avgfunc(psds[:, 9:14, :], axis=0), avgfunc(psds[:, 14:16, :], axis=0)]), axis=0)
    fg.fit(freqs, np.reshape(occipital_psds, (1, len(occipital_psds))), f_range)
    occipital_foof = fg.get_fooof(0, True)

    # Left Hemisphere
    ''' نیم کره چپ و راست برای این تحلیل سیگنال مهم نیست'''
    ''' این ایندکس ها رندم هستند صرفا برای اینکه فانکشن کار کند'''
    left_index = np.hstack([channels[0], channels[3], channels[5:9], channels[14:18], channels[23:27],
                            channels[23:27], channels[23:27], channels[2:5], channels[5]])
    left_psds = avgfunc(np.vstack([avgfunc(psds[:, 0, :], axis=0), avgfunc(psds[:, 3, :], axis=0),
                                   avgfunc(avgfunc(psds[:, 5:9, :], axis=0), axis=0),
                                   avgfunc(avgfunc(psds[:, 14:18, :], axis=0), axis=0),
                                   avgfunc(avgfunc(psds[:, 23:27, :], axis=0), axis=0),
                                   avgfunc(avgfunc(psds[:, 22:27, :], axis=0), axis=0),
                                   avgfunc(avgfunc(psds[:, 13:16, :], axis=0), axis=0),
                                   avgfunc(avgfunc(psds[:, 1:6, :], axis=0), axis=0),
                                   avgfunc(psds[:, 5, :], axis=0)]), axis=0)
    fg.fit(freqs, np.reshape(left_psds, (1, len(left_psds))), f_range)
    left_foof = fg.get_fooof(0, True)

    # Right Hemisphere
    right_index = np.hstack([channels[2], channels[4], channels[10:14], channels[19:23], channels[1:5],
                             channels[5:12], channels[3:12], channels[1:3], channels[4]])
    right_psds = avgfunc(np.vstack([avgfunc(psds[:, 2, :], axis=0), avgfunc(psds[:, 4, :], axis=0),
                                    avgfunc(avgfunc(psds[:, 10:14, :], axis=0), axis=0),
                                    avgfunc(avgfunc(psds[:, 19:23, :], axis=0), axis=0),
                                    avgfunc(avgfunc(psds[:, 2:12, :], axis=0), axis=0),
                                    avgfunc(avgfunc(psds[:, 13:15, :], axis=0), axis=0),
                                    avgfunc(avgfunc(psds[:, 1:3, :], axis=0), axis=0),
                                    avgfunc(avgfunc(psds[:, 2:5, :], axis=0), axis=0),
                                    avgfunc(psds[:, 3, :], axis=0)]), axis=0)
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
    aperiodic_mode = 'fixed'
    for region in regions:
        spec_parkinson = []
        foof_parkinson = []
        dterend_parkinson = []
        dterend_healthy = []
        curve_parkinson = []
        spec_healthy = []
        foof_healthy = []
        curve_healthy = []
        for index, subj in enumerate(data1):
            freqs = data1[f'{subj}']['Frontal'].freqs
            real_spec = data1[f'{subj}'][f'{region}'].power_spectrum
            foof = data1[f'{subj}'][f'{region}'].fooofed_spectrum_
            if aperiodic_mode == 'fixed':
                curve = data1[f'{subj}'][f'{region}'].aperiodic_params_[0] \
                        - np.log10(0.00 + freqs ** data1[f'{subj}'][f'{region}'].aperiodic_params_[1])
            else:
                curve = data1[f'{subj}'][f'{region}'].aperiodic_params_[0] \
                        - np.log10(0.00 + freqs ** data1[f'{subj}'][f'{region}'].aperiodic_params_[2])
            spec_parkinson.append(real_spec)
            foof_parkinson.append(foof)
            curve_parkinson.append(curve)
            dterend_parkinson.append(foof - curve)
        # signals['parkinson'][f'{region}']['real_spec'] = np.array(spec_parkinson)
        signals['parkinson'][f'{region}'] = np.array(dterend_parkinson)
        # signals['parkinson'][f'{region}']['curve'] = np.array(curve_parkinson)

        for index, subj in enumerate(data2):
            real_spec_sec = data2[f'{subj}'][f'{region}'].power_spectrum
            foof_sec = data2[f'{subj}'][f'{region}'].fooofed_spectrum_
            if aperiodic_mode == 'fixed':
                curve_sec = data2[f'{subj}'][f'{region}'].aperiodic_params_[0] \
                            - np.log10(0.00 + freqs ** data2[f'{subj}'][f'{region}'].aperiodic_params_[1])
            else:
                curve_sec = data2[f'{subj}'][f'{region}'].aperiodic_params_[0] \
                            - np.log10(0.00 + freqs ** data2[f'{subj}'][f'{region}'].aperiodic_params_[2])
            spec_healthy.append(real_spec_sec)
            foof_healthy.append(foof_sec)
            curve_healthy.append(curve_sec)
            dterend_healthy.append(foof_sec - curve_sec)
        signals['healthy'][f'{region}'] = np.array(dterend_healthy)
        # signals['healthy'][f'{region}']['real_spec'] = np.array(spec_healthy)
        # signals['healthy'][f'{region}']['curve'] = np.array(curve_healthy)
        fig, ax1 = plt.subplots(1, 1)
        plt.grid(True)
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

''' Getting Significant Exponent and Offset values'''


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


def get_area(data, f_range, srate, save_root):
    areas = dict()
    # harmonic = dict()
    exper = ['parkinson', 'healthy']
    regions = ['Frontal', 'Central', 'Occipital', 'Right', 'Left', 'AllScalp']
    bands = ['delta', 'alpha', 'beta']
    features = ['peak', 'CF']
    n_fft = srate / (20 * srate)
    # Harmonic
    # for ex in exper:
    #     harmonic[ex] = dict()
    #     for region in regions:
    #         harmonic[ex][region] = dict()
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
        a1 = data[0]['parkinson'][region]
        a2 = data[0]['healthy'][region]
        freqs = data[1]

        '''
        delta = 1-4
        alpha = 8-12
        beta = 14-25
        freq_index = (f_interest - f_min) * 2 
        '''

        if 1 in freqs:
            delta_low = np.array(np.where(freqs == 1))[0][0]
            delta_high = np.array(np.where(freqs == 4))[0][0]
            peak_delta1 = np.max(a1[:, delta_low:delta_high], axis=1)
            peak_delta2 = np.max(a2[:, delta_low:delta_high], axis=1)
            areas['parkinson'][region]['delta']['peak'] = np.round(peak_delta1, 4)
            areas['healthy'][region]['delta']['peak'] = np.round(peak_delta2, 4)
        else:
            areas['parkinson'][region]['delta']['peak'] = np.zeros((1, len(a1)))[0]
            areas['healthy'][region]['delta']['peak'] = np.zeros((1, len(a2)))[0]

        alpha_low = np.array(np.where(freqs == 7))[0][0]
        alpha_high = np.array(np.where(freqs == 12))[0][0]
        alpha_range = np.arange(5, 12 + n_fft, n_fft)

        # beta_low = np.array(np.where(freqs == 14))[0][0]
        # beta_high = np.array(np.where(freqs == 30))[0][0]
        # beta_range = np.arange(14, 30 + n_fft, n_fft)

        # Peak Amplitude
        peak_alpha1 = np.max(a1[:, alpha_low:alpha_high + 1], axis=1)
        peak_alpha2 = np.max(a2[:, alpha_low:alpha_high + 1], axis=1)
        freq_alpha1 = np.argmax(a1[:, alpha_low:alpha_high + 1], axis=1)
        freq_alpha2 = np.argmax(a2[:, alpha_low:alpha_high + 1], axis=1)
        # print('freq_alpha1', freq_alpha1, 'alpha range', alpha_range, 'alpha', alpha_low, alpha_high)
        freq_alpha1 = alpha_range[freq_alpha1]
        freq_alpha2 = alpha_range[freq_alpha2]
        # beta
        # peak_beta1 = np.max(a1[:, beta_low:beta_high + 1], axis=1)
        # peak_beta2 = np.max(a2[:, beta_low:beta_high + 1], axis=1)
        # freq_beta1 = np.argmax(a1[:, beta_low:beta_high + 1], axis=1)
        # freq_beta2 = np.argmax(a2[:, beta_low:beta_high + 1], axis=1)
        # freq_beta1 = beta_range[freq_beta1]
        # freq_beta2 = beta_range[freq_beta2]

        areas['parkinson'][region]['alpha']['peak'] = np.round(peak_alpha1, 4)
        areas['healthy'][region]['alpha']['peak'] = np.round(peak_alpha2, 4)
        areas['parkinson'][region]['alpha']['CF'] = freq_alpha1
        areas['healthy'][region]['alpha']['CF'] = freq_alpha2

        # areas['parkinson'][region]['beta']['peak'] = np.round(peak_beta1, 4)
        # areas['healthy'][region]['beta']['peak'] = np.round(peak_beta2, 4)
        # areas['parkinson'][region]['beta']['CF'] = freq_beta1
        # areas['healthy'][region]['beta']['CF'] = freq_beta2

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

        # fig2 = plt.figure()
        # plt.grid(True)
        # plt.plot(freq_alpha2, freq_beta2, 'o')
        # plt.plot(line_graph_xaxis, line_graph_yaxis)
        # plt.xlabel(f'Alpha\nalpha/beta harmonic ratio = %{(harmonic_detect(freq_alpha2, freq_beta2)) * 100}')
        # plt.ylabel(f'Beta')
        # plt.title(f'Healthy alpha/beta harmonic over {region}')
        # plt.subplots_adjust(left=0.15, bottom=0.26, right=None, top=None, wspace=None, hspace=None)
        # plt.savefig(save_root + f'/Healthy alpha-beta correlation over {region}.png')

    return areas#, harmonic


# def harmonic_detect(alpha, beta):
#     harmon = 0
#     for i in range(len(alpha)):
#         # print('Len = ', len(alpha), '\n alpha = ', alpha, '\nbeta = ', beta)
#         if alpha[i] == beta[i] / 2 or alpha[i] == (beta[i] / 2) + 0.1 or alpha[i] == (beta[i] / 2) - 0.1:
#             harmon += 1
#     proportion = harmon / len(alpha)
#     return proportion
