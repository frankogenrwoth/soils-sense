# SOIL SENSE

**Soil Sense** is an intelligent soil moisture monitoring web application designed to help farmers optimize irrigation practices through real-time data analysis and machine learning predictions. This project serves as the Artificial Intelligence research project for Year 3 students at Makerere University College of Computing and Information Sciences (COCIS).

## Features

### Core Functionality
- **Real-time Soil Moisture Monitoring**: Track soil moisture levels from IoT devices, CSV uploads, or manual input
- **Machine Learning Predictions**: AI-powered forecasting of future soil moisture levels
- **Smart Irrigation Recommendations**: Automated suggestions based on crop requirements and predictions
- **Interactive Dashboards**: Visualize data with charts and graphs using Chart.js
- **Role-based Access Control**: Secure user management with Admin, Technician, and Farmer roles
- **Comprehensive Reporting**: Generate and download reports in PDF/Excel formats

### User Management
- User registration and authentication
- Password reset and profile management
- Role-based access control (Admin, Technician, Farmer)
- Secure session management

### Data Management
- Store and retrieve soil moisture readings
- Filter data by location, date range, and crop type
- Historical data analysis and trending
- Data export capabilities

### AI/ML Integration
- Predictive modeling for soil moisture levels
- Model retraining capabilities for admins
- Weather forecast integration
- Automated threshold-based alerts

## 🛠 Technology Stack

### Backend
- **Django 5.2.4** - Web framework
- **Django REST Framework** - API development
- **Python 3.x** - Programming language
- **SQLite3** - Database

### Frontend
- **HTML/CSS/JavaScript** - Frontend technologies
- **Chart.js** - Data visualization
- **Tailwind CSS** - UI framework

### Machine Learning
- **scikit-learn** - ML algorithms
- **Pandas/NumPy** - Data processing

### DevOps
- **UbiOps** - Model inference
- **Git** - Version control

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/<username>/soil-sense.git
   cd soil-sense
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser**
   ```bash
   python manage.py createsuperuser
   ```

7. **Run the development server**
   ```bash
   python manage.py runserver
   ```

8. **Access the application**
   - Open your browser and go to `http://127.0.0.1:8000/`
   - Admin panel: `http://127.0.0.1:8000/admin/`

## Project Structure

```
soilsense/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── .env.example             # Environment variables template
├── .gitignore              # Git ignore rules
└── soilsense/              # Main Django project
    ├── __init__.py
    ├── settings.py         # Django settings
    ├── urls.py            # Main URL configuration
    ├── wsgi.py           # WSGI configuration
    └── asgi.py           # ASGI configuration
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory with the following variables:

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Database Configuration
The database comes already set up for the light nature of the project we use sqlite3 which has a well documented api for django

## Usage

### For Farmers
1. Register an account with farmer role
2. Add your farm locations and crop information
3. View real-time soil moisture data
4. Receive irrigation recommendations
5. Download reports for record keeping

### For Technicians
1. Access system with technician credentials
2. Monitor multiple farm locations
3. Configure sensor thresholds
4. Generate technical reports

### For Administrators
1. Manage user accounts and permissions
2. Upload and retrain ML models
3. Configure system-wide settings
4. Access comprehensive analytics

## Machine Learning Features

### Prediction Model
- **Input Parameters**: Location, current soil moisture, temperature, humidity
- **Output**: Predicted soil moisture levels for next 7-30 days
- **Model Types**: Linear regression, Random Forest, Neural Networks
- **Retraining**: Admin can upload new training data and retrain models

### Data Sources
- IoT sensor data
- Weather API integration
- Historical soil moisture records
- Crop-specific water requirements

## API Documentation

The application provides RESTful APIs for:
- User authentication and management
- Soil moisture data CRUD operations
- ML prediction endpoints
- Report generation
- Alert management

### Example API Endpoints
```
POST /api/auth/login/          # User login
GET  /api/moisture/           # Get moisture data
POST /api/predict/            # Get ML predictions
GET  /api/reports/            # Generate reports
```

## Testing

Run the test suite:
```bash
python manage.py test
```

Run with coverage:
```bash
coverage run --source='.' manage.py test
coverage report
```

## Deployment

### Production Setup
1. Set `DEBUG=False` in settings
2. Configure production database
3. Set up static file serving


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Team

**Year 3 Students - Makerere University COCIS**
- [Student Name 1] - Backend Development
- [Student Name 2] - Frontend Development  
- [Student Name 3] - Machine Learning
- [Student Name 4] - Database Design

## Support

For support and questions:
- Email: soilsense@mak.ac.ug
- GitHub Issues: [Create an issue](https://github.com/yourusername/soilsense/issues)
- Documentation: [Wiki](https://github.com/yourusername/soilsense/wiki)

## Version History

- **v1.0.0** - Initial release with basic soil moisture monitoring

---

**Made with ❤️ by Makerere University BSSE Group P Students**
