#!/bin/bash
# Setup script for Financial Signal System

echo "🚀 Setting up Financial Signal Generation System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment (Linux/macOS)
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated (Linux/macOS)"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
    echo "✅ Virtual environment activated (Windows)"
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/cache
mkdir -p data/signals

# Create .gitkeep files
touch data/cache/.gitkeep
touch data/signals/.gitkeep

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🔧 Next Steps:"
echo "1. Create .env file with your API keys (see .env.template)"
echo "2. Install TA-Lib library (platform-specific)"
echo "3. Run the application: python run.py"
echo ""
echo "📚 TA-Lib Installation:"
echo "   Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
echo "   macOS: brew install ta-lib"
echo "   Linux: sudo apt-get install build-essential && install from source"
echo ""
echo "🌐 Access your dashboard at: http://localhost:8501"
