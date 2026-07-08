module.exports = {
  apps: [
    {
      name: 'dashboardaq-web',
      cwd: '/home/yusuf/Monitoring-Kualitas-Udara-AQindex',
      script: 'bun',
      args: 'run start',
      env: {
        PORT: 3000,
        PYTHON_PATH: '/home/yusuf/Monitoring-Kualitas-Udara-AQindex/ml_model/venv/bin/python3',
      },
    },
    {
      name: 'dashboardaq-watcher',
      cwd: '/home/yusuf/Monitoring-Kualitas-Udara-AQindex/ml_model',
      script: 'live_forecast_watcher.py',
      interpreter: '/home/yusuf/Monitoring-Kualitas-Udara-AQindex/ml_model/venv/bin/python3',
    },
  ],
};
