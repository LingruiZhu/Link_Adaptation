# Link adaptation Simulation
 ## Assumptions made
  in our simulation, the blow assumptions are made: (1) CQI is perfectly known.  (2) TBS is calculated arbitrarily (3) SINR is linear to CQI.
 ## Structrue
  In this repository, LutLinkSimulation --> LutEnvironment <--> La_agent or any other LA algorithm
 ## LUT data format
  The relationships among, bler, MCS, SINR/CQI are stored in the .npy file. Each .npy file contains a dictionary with two layers: the first one is SINR, each containing a list indicating the relationships between MCS and BLER.
  Subcarrier spacing: 30kHz.
 ## OLLA
  At first, estimate SINR according to CQI. Then, find the closest sinr stored in LUT data. Lastly, calculate reward accrding to data from LUT.
  Things may be added in the future: not just directly use LUT data directly, but also use regression to estimate blers for the current SINR.
