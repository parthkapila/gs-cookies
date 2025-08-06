import React, { useState, useEffect, useRef } from "react";
import { FaHome, FaInfoCircle, FaQuestionCircle, FaChevronDown } from "react-icons/fa";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from "recharts";
import "./index.css";

// Base URL for backend API
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:5001"; //"https://gsci-backend.onrender.com";

/** Helper: convert period integer (e.g., 1, 2, 3...) to actual year (2019 + period) */
function periodToYear(period) {
  return 2019 + period;
}

/** Helper: color palette for Recharts */
function getColor(idx) {
  const palette = [
    "#8884d8",
    "#82ca9d",
    "#ffc658",
    "#d0ed57",
    "#a4de6c",
    "#8dd1e1",
    "#d88884",
    "#ad8de1",
    "#84d8a4",
    "#e1cf8d",
  ];
  return palette[idx % palette.length];
}

/** Helper: format tooltip values to 2 decimal places */
function formatTooltipValue(value) {
  if (typeof value === 'number') {
    return value.toFixed(2);
  }
  return value;
}

/** Custom tooltip formatter for Recharts */
function customTooltipFormatter(value, name, props) {
  return [formatTooltipValue(value), name];
}

function getTrendline(data, xKey, yKey) {
  if (!data || data.length < 2) return [];

  const n = data.length;
  const sumX = data.reduce((sum, d) => sum + d[xKey], 0);
  const sumY = data.reduce((sum, d) => sum + d[yKey], 0);
  const meanX = sumX / n;
  const meanY = sumY / n;

  let numerator = 0;
  let denominator = 0;

  data.forEach(d => {
    numerator += (d[xKey] - meanX) * (d[yKey] - meanY);
    denominator += (d[xKey] - meanX) ** 2;
  });

  const slope = numerator / denominator;
  const intercept = meanY - slope * meanX;

  const minX = Math.min(...data.map(d => d[xKey]));
  const maxX = Math.max(...data.map(d => d[xKey]));

  return [
    {
      [xKey]: minX,
      [yKey]: slope * minX + intercept
    },
    {
      [xKey]: maxX,
      [yKey]: slope * maxX + intercept
    }
  ];
}

/** Cookie info for the ReturningTroopPage predictions grid */
// Remove the cookies array with name/image mapping

/** Custom Dropdown Component that matches existing input styling */
function CustomDropdown({ 
  value, 
  onChange, 
  options, 
  placeholder,
  displayKey = null,
  valueKey = null
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const dropdownRef = useRef(null);

  // Filter options based on search term - show options that contain the search term anywhere
  const filteredOptions = options.filter(option => {
    if (displayKey && valueKey) {
      // For objects, search in both display and value keys
      const displayText = option[displayKey]?.toString() || '';
      const valueText = option[valueKey]?.toString() || '';
      return displayText.includes(searchTerm) || valueText.includes(searchTerm);
    } else {
      // For simple strings/numbers
      return option.toString().includes(searchTerm);
    }
  });

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
        setSearchTerm("");
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (option) => {
    // If we have valueKey, use that value, otherwise use the option itself
    const selectedValue = valueKey ? option[valueKey] : option;
    onChange(selectedValue);
    setIsOpen(false);
    setSearchTerm("");
  };

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setSearchTerm("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (searchTerm) {
        // Check if there's an exact match first
        const exactMatch = options.find(option => 
          option.toString() === searchTerm
        );
        if (exactMatch) {
          handleSelect(exactMatch);
        } else if (filteredOptions.length > 0) {
          // If no exact match but there are filtered options, select the top one
          handleSelect(filteredOptions[0]);
        } else {
          // No exact match and no filtered options, keep dropdown open with "No results found"
          setIsOpen(true);
        }
      } else {
        setIsOpen(!isOpen);
      }
    }
  };

  return (
    <div className="custom-dropdown" ref={dropdownRef}>
      <input
        type="text"
        value={isOpen ? searchTerm : (value || "")}
        onChange={(e) => {
          setSearchTerm(e.target.value);
          if (!isOpen) setIsOpen(true);
        }}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        onFocus={() => setIsOpen(true)}
        onBlur={() => {
          // Delay closing to allow for option selection
          setTimeout(() => setIsOpen(false), 200);
        }}
      />
      <FaChevronDown 
        className={`dropdown-arrow ${isOpen ? 'rotated' : ''}`}
        onClick={toggleDropdown}
      />
      
      {isOpen && (
        <div className="dropdown-options">
          <div className="dropdown-options-container">
            {filteredOptions.length > 0 ? (
              filteredOptions.map((option, index) => (
                <div
                  key={index}
                  className="dropdown-option"
                  onMouseDown={() => handleSelect(option)}
                >
                  {displayKey && valueKey ? 
                    `${option[valueKey]} - ${option[displayKey]}` : 
                    option
                  }
                </div>
              ))
            ) : (
              <div className="dropdown-option no-results">
                No results found
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}



/* ------------------------------------------------------------------
   PAGE 1: LandingPage -> user picks "New Troop" or "Returning Troop"
   ------------------------------------------------------------------ */
   function LandingPage({ onNewTroop, onReturningTroop, onManual, onAbout, onFaq, onHome }) {
    return (
      <div className="main-container">
        <div className="background"></div>
        <div className="overlay"></div>
        <header className="header">
          <div className="logo-row">
            <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
          </div>
          <nav className="nav-links">
            <div className="nav-link" onClick={onHome}><FaHome /></div>
            <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
            <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
          </nav>
        </header>
        <h1 className="title">Sales Prediction Platform</h1>
        <p className="subtitle">Forecasting Sales, One Cookie at a Time</p>
        <div className="input-container">
          <p>Welcome! Please select your troop type below.</p>
          <button className="predict-button" onClick={onReturningTroop}>Returning Troop</button>
          <button className="predict-button" onClick={onNewTroop}>New Troop</button>
        </div>
      </div>
    );
  }

  function ManualPage({ onBack, onAbout, onFaq, onHome }) {
    const [isDetailsVisible, setIsDetailsVisible] = useState(false);
    const [lightMode, setLightMode] = useState(false);
  
    const handleShowDetails = () => {
      setIsDetailsVisible(true);
      setTimeout(() => {
        const section = document.getElementById("details");
        section?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    };
  
    const toggleTheme = () => {
      setLightMode((prev) => !prev);
    };
    return (
      <div className={`main-container ${lightMode ? "light-mode" : ""}`}>
        <div className="background"></div>
        <div className="overlay"></div>
        <header className="header">
          <div className="logo-row">
            <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
          </div>
          <nav className="nav-links">
            <div className="nav-link" onClick={onHome}><FaHome /></div>
            <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
            <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
          </nav>
        </header>
  
        <div className="title">Cookie Forecasting Manual</div>
        <div className="manual-content">
                  <div className="title">Cookie Forecasting Manual</div>

          <div className="content">
            <p>Explore how predictive analytics can help Girl Scouts maximize cookie sales.</p>
            <p>Click the button below to explore data-driven insights, model findings, and outcomes.</p>
            <button className="view-details" onClick={handleShowDetails}>View Details</button>
            <button className="toggle-theme" onClick={toggleTheme}>Toggle Theme</button>
          </div>

          {isDetailsVisible && (
            <div className="details-section" id="details">
              <h2>Business Problem</h2>
              <p>Every year, thousands of Girl Scouts rely on cookie sales for fundraising. The current forecast method—based solely on the previous year's numbers—explains only 70% of sales variability. This limitation often leads to missed revenue opportunities or excess stock. By leveraging advanced predictive models, we can bridge this gap and unlock new opportunities for growth, efficiency, and increased fundraising success.</p>

              <h2>Key Benefits</h2>
              <ul>
                <li>Improved inventory planning and cost savings</li>
                <li>Enhanced Marketing Strategies</li>
                <li>IncreasedTroop Engagement</li>
                <li>Data Driven Decision Making</li>
              </ul>

              <h2>Analytical Problem</h2>
              <p><strong>Analytical Context:</strong> The context involves analyzing historical sales data and external factors to improve forecasting accuracy by troop and cookie type.</p>
              <p><strong>Challenges:</strong> Challenges include high variance in sales across troops and regions, impact from external factors like weather and local events, and ensuring model reliability for troop leaders.</p>
              <p><strong>Solution Focus:</strong> The solution focuses on implementing machine learning models to enhance forecasting accuracy and optimize inventory management.</p>

              <h2>Research Questions</h2>
              <ul>
                <li>Can machine learning models effectively integrate historical sales data, troop participation rates to improve the accuracy of cookie sales forecasts beyond traditional methods?</li>
                <li>How can insights from predictive models be used to optimize inventory management and marketing strategies for Girl Scout troops, ensuring that each troop meets demand without costly surplus or shortages?</li>
              </ul>

              <h2>Data Dictionary</h2>
              <img src="ch5.png" alt="Data Results" />
              <img src="Ch1.png" alt="Data Chart" />

              <h2>Model Selection</h2>
              <strong>ASSUMPTIONS:</strong>
              <ul>
                <li>The model assumes past sales patterns are predictive of future sales performance.</li>
                <li>No major disruptions in cookie availability, troop operations, or supply chains are expected.</li>
              </ul>
              <strong>LIMITATION:</strong>
              <ul>
                <li>The model may underperform for troops with limited or erratic historical data.</li>
              </ul>
              <strong>SPLITTING:</strong>
              <ul>
                <li>The dataset is grouped by troop ID and cookie type. For each group, the data is split by year into Training (2020–2023) and Testing (2024).</li>
                <li>Cluster-based modeling is applied within each location and cookie type, allowing for more personalized predictions.</li>
              </ul>
              <strong>BASE MODEL APPROACHES:</strong>
              <ul>
                <li><strong>SIO Model:</strong> Uses last year's sales and troop participation to estimate.</li>
                <li><strong>AVG Model:</strong> Averages past sales from 2021-2023 to predict 2024. However, these models had higher RMSE values, indicating significant prediction errors.</li>
              </ul>
              <img src="Ch2.jpg" alt="Base Model Chart" />
              <p>To improve accuracy, we built a Hybrid Multi-Model system that automatically selects the best among Clustered Ridge Regression, Troop-Level Ridge Regression, Linear Regression, SIO & Average, and Location-Level Ridge Regression. Each troop-cookie prediction uses the method with the lowest error, dynamically chosen based on past performance. We tested other models and found the Hybrid had the best performance.</p>
              <img src="ch4.jpg" alt="Model Selection Chart" />

              <h2>Validation</h2>
              <p><strong>Confidence in Predictions:</strong> Model predictions are validated using cross-validation (CV) within each training group to optimize regularization strength (λ), minimizing overfitting and ensuring stable performance across troops and cookie types.</p>
              <p><strong>Dynamic Error-Based Method Selection:</strong> For each troop-cookie pair, the model dynamically selects the prediction method with the lowest expected error (MSE), based on past performance. This approach ensures predictions are customized and evidence-driven.</p>
              <p><strong>Robustness Across Segments:</strong> The use of clustering + Ridge + fallback heuristics allows the model to adapt to sparse, dense, and even noisy troop histories—leading to consistent accuracy improvements over baseline models.</p>

              <h2>Final Model Results</h2>
              <img src="metric.jpg" alt="Metrics" />

              <h2>Key Findings</h2>
              <p>Our Hybrid Multi-Model approach, which blends Ridge Regression (with CV), linear models, and PGA heuristics, achieved the highest R² and lowest error metrics, highlighting its adaptability and accuracy. The model dynamically selects the best prediction method (from 6 options) for each troop-cookie pair, based on historical performance and data quality.</p>
              <p>By improving prediction accuracy by 1.35 cases per troop per cookie type compared to the SIO tool, our model provided a significant advantage in planning and inventory. Scaled across 1,401 troops, 8 cookie types, and 12 boxes per case, this translates to a potential impact of over 181,000 boxes—equivalent to more than</p>
              <img src="KF.JPG" alt="Key Findings Graphic" style={{ marginTop: '20px', borderRadius: '10px' }} />
              <p>This enables GS Indiana to optimize inventory, reduce over-ordering, and boost revenue for smarter, more profitable decisions.</p>

              <h2>Model Life Cycle</h2>
              <img src="ch6.jpg" alt="Model Life Cycle" />

              <div className="pdf-link">
                <a href="SC.pdf" target="_blank" rel="noopener noreferrer">View Full Poster PDF</a>
              </div>
            </div>
          )}

        </div>
      </div>
    );
  }
  
/* ------------------------------------------------------------------
   PAGE 2: NewTroopSearchPage -> user enters Service Unit # and sees suggestions (non-clickable)
   ------------------------------------------------------------------ */
function NewTroopSearchPage({ onSearch, onBack, onAbout, onFaq, onHome }) {
  const [suInput, setSuInput] = useState("");
  const [numGirls, setNumGirls] = useState("");
  const [error, setError] = useState("");
  const [suOptions, setSuOptions] = useState([]);

  // Fetch all Service Unit options on mount
  useEffect(() => {
    const fetchSUOptions = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/su_search?q=`);
        const data = await res.json();
        // Store full data with both SU_Num and SU_Name
        setSuOptions(data);
      } catch (err) {
        console.error("Error fetching Service Unit options:", err);
      }
    };
    fetchSUOptions();
  }, []);

  const handleSearch = async () => {
    if (!/^\d+$/.test(suInput)) {
      setError("Please enter a valid Service Unit number (digits only).");
      return;
    }
    if (!numGirls || isNaN(numGirls)) {
      setError("Please enter a valid number of girls.");
      return;
    }
    const validatedGirls = Math.max(0, Math.min(250, Number(numGirls)));
    setError("");
    
    // Debug logging
    console.log("SU Input:", suInput, "Type:", typeof suInput);
    console.log("SU Options:", suOptions.slice(0, 3));
    console.log("SU Options types:", suOptions.slice(0, 3).map(opt => ({ SU_Num: opt.SU_Num, type: typeof opt.SU_Num })));
    
    // Find the selected Service Unit from the options
    // Handle both string and number types for robust comparison
    const selectedServiceUnit = suOptions.find(option => 
      option.SU_Num.toString() === suInput.toString() || 
      option.SU_Num === parseInt(suInput)
    );
    
    console.log("Selected Service Unit:", selectedServiceUnit);
    
    if (!selectedServiceUnit) {
      setError("No Service Unit found with that number.");
      return;
    }
    
    onSearch(selectedServiceUnit.SU_Num, selectedServiceUnit.SU_Name, validatedGirls);
  };

  return (
    <div className="main-container">
      <div className="background"></div>
      <div className="overlay"></div>
      <header className="header">
        <div className="logo-row">
            <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
        </div>
        <nav className="nav-links">
          <div className="nav-link" onClick={onHome}><FaHome /></div>
          <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
          <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
        </nav>
      </header>
      <h1 className="title">New Troop Service Unit Search</h1>
      <p className="subtitle">Enter your Service Unit number and Number of Girls</p>
             <div className="input-container">
         <div className="input-box">
           Service Unit:{" "}
           <CustomDropdown
             value={suInput}
             onChange={setSuInput}
             options={suOptions}
             placeholder="e.g. 153"
             displayKey="SU_Name"
             valueKey="SU_Num"
           />
         </div>
        <div className="input-box">
          Number of Girls:{" "}
          <input
            type="number"
            max="250"
            value={numGirls}
            onChange={(e) => setNumGirls(e.target.value)}
            placeholder="e.g. 25"
          />
        </div>
        <div style={{ display: "flex", gap: "20px", marginTop: "20px" }}>
          <button className="predict-button" onClick={onBack}>Back</button>
          <button className="predict-button" onClick={handleSearch}>Search</button>
        </div>
      </div>
      {/* Suggestions now rendered via datalist; removed legacy list */}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------
   PAGE 3: NewTroopAnalyticsPage -> show Service Unit analytics (charts, scatter, etc.)
   Includes:
     - A new input for the question: "How many girls do you think will sell?"
     - Vertical highlighting (via ReferenceLine) in each scatter chart at that value.
     - Updated chart title for total average cases sold.
   ------------------------------------------------------------------ */
      function NewTroopAnalyticsPage({ suNumber, suName, initialNumGirls = "", onBack, onAbout, onFaq, onHome }) {
    const [girlsData, setGirlsData] = useState([]);
    const [salesData, setSalesData] = useState([]);
    const [scatterData, setScatterData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [predictedGirls, setPredictedGirls] = useState(Math.max(0, Math.min(250, initialNumGirls)));
    const [suPredictions, setSuPredictions] = useState([]);
    const [predictionsLoading, setPredictionsLoading] = useState(false);
         // State for percentage field
    const [percentage, setPercentage] = useState("50");
    const [percentageError, setPercentageError] = useState("");
    const [appliedPercentage, setAppliedPercentage] = useState(50); // State for the percentage applied to predictions
    
    const cookieTypes = Array.from(new Set(scatterData.map((d) => d.canonical_cookie_type)));

    useEffect(() => {
      const fetchHistory = async () => {
        if (!suNumber) return;
        setLoading(true);
        setError("");
        try {
          const historyRes = await fetch(`${API_BASE}/api/su_history/${suNumber}`);
          const history = await historyRes.json();
          setGirlsData(history.girlsByYear || []);
          setSalesData(history.salesByYear || []);
          setScatterData(history.scatterData || []);
        } catch (err) {
          console.error("Error fetching Service Unit history:", err);
          setError("Could not fetch Service Unit history.");
        } finally {
          setLoading(false);
        }
      };
      fetchHistory();
    }, [suNumber]);

    // Function to fetch Service Unit predictions using a given number of girls
    const fetchSUPredictions = async (girlsVal) => {
      setPredictionsLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/su_predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            su_number: suNumber,
            num_girls: Number(girlsVal)
          }),
        });
        const data = await res.json();
        
        // Format the data the same way as ReturningTroopAnalyticsPage
        const formatted = {};
        data.forEach((d) => {
          const key = d.cookie_type;
          formatted[key] = {
            predictedCases: d.predicted_cases,
            last_year_based_prediction: d.last_year_based_prediction,
            last_year_sales: d.last_year_sales,
            imageUrl: d.image_url,
            source: d.source
          };
        });
        setSuPredictions(formatted);
      } catch (err) {
        console.error("Error fetching Service Unit predictions:", err);
      } finally {
        setPredictionsLoading(false);
      }
    };

    // Fetch initial predictions **once** when the Service Unit loads and we have an initial girls value.
    const initFetched = React.useRef(false);
    useEffect(() => {
      if (!initFetched.current && suNumber && predictedGirls && !isNaN(predictedGirls)) {
        fetchSUPredictions(predictedGirls);
        initFetched.current = true; // ensure we only run once for this Service Unit load
      }
    }, [suNumber, predictedGirls]);

    const handleUpdatePredictions = () => {
      if (predictedGirls && !isNaN(predictedGirls)) {
        const clamped = Math.max(0, Math.min(250, Number(predictedGirls)));
        setPredictedGirls(clamped);
        fetchSUPredictions(clamped);
        
        // Apply the percentage when button is clicked
        if (percentage && !isNaN(parseFloat(percentage))) {
          const percentageValue = parseFloat(percentage);
          if (percentageValue >= 0 && percentageValue <= 100) {
            setAppliedPercentage(percentageValue);
          }
        } else {
          setAppliedPercentage(50); // Default to 50% if no valid percentage
        }
      }
    };

    const handlePercentageChange = (e) => {
      const value = e.target.value;
      setPercentage(value);
      
      // Clear error when user starts typing
      if (percentageError) {
        setPercentageError("");
      }
      
      // Validate percentage range
      const numValue = parseFloat(value);
      if (value !== "" && (isNaN(numValue) || numValue < 0 || numValue > 100)) {
        setPercentageError("Please enter a value between 0-100");
      } else {
        setPercentageError("");
      }
    };

    // Function to calculate adjusted predicted cases based on percentage
    const getAdjustedPredictedCases = (originalCases) => {
      if (!appliedPercentage || isNaN(appliedPercentage)) {
        return originalCases;
      }
      
      if (appliedPercentage < 0 || appliedPercentage > 100) {
        return originalCases;
      }
      
      // Convert percentage to multiplier (e.g., 50% = 0.5, 150% = 1.5)
      const multiplier = appliedPercentage / 100;
      return originalCases * multiplier;
        };

    // Function to calculate 100% predicted cases (for comparison)
    const get100PercentPredictedCases = (originalCases) => {
      return originalCases;
    };

    return (
      <div className="main-container">
        <div className="background"></div>
        <div className="overlay"></div>
        <header className="header">
          <div className="logo-row">
            <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
          </div>
          <nav className="nav-links">
            <div className="nav-link" onClick={onHome}><FaHome /></div>
            <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
            <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
          </nav>
        </header>
        <h1 className="title">Service Unit Dashboard</h1>
        <p className="subtitle">
          Showing analytics for Service Unit #{suNumber}{suName ? ` - ${suName}` : ""}
        </p>
  
        <div className="input-container" style={{ marginBottom: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "20px", marginBottom: "10px" }}>
            <div style={{ display: "flex", alignItems: "center" }}>
              <label style={{ marginRight: "10px" }}>
                How many girls do you think will sell?
              </label>
              <input
                type="number"
                value={predictedGirls}
                onChange={(e) => {
                  const val = e.target.value;
                  setPredictedGirls(val);
                }}
                placeholder="Enter a number"
                style={{ width: "80px" }}
                max="250"
                min="0"
              />
            </div>
            <div style={{ display: "flex", alignItems: "center" }}>
              <label style={{ marginRight: "10px" }}>Percentage (%):</label>
              <input
                type="number"
                value={percentage}
                onChange={handlePercentageChange}
                style={{ width: "80px" }}
                placeholder="0-100"
                step="0.1"
              />
            </div>
          </div>
          {percentageError && (
            <div style={{ color: "red", fontSize: "14px", marginBottom: "10px" }}>
              {percentageError}
            </div>
          )}
          <div style={{ display: "flex", gap: "20px" }}>
            <button className="predict-button" onClick={handleUpdatePredictions}>
              Update Predictions
            </button>
            <button className="predict-button" onClick={onBack}>Back</button>
          </div>
        </div>

        {/* Cookie Predictions Section */}
        <div style={{ fontSize: "18px", marginBottom: "30px" }}>
          <div className="predictions">PREDICTIONS</div>
          {predictionsLoading ? (
            <div className="spinner" style={{ margin: '40px auto' }}></div>
          ) : (
            Object.keys(suPredictions).length > 0 && (
              <div className="cookie-grid" style={{ background: "none", padding: "20px" }}>
                {Object.entries(suPredictions).map(([cookieName, pred]) => (
                  <div key={cookieName} className="cookie-box" style={{ fontSize: "18px" }}>
                    <img src={pred.imageUrl} alt={cookieName} />
                    <div className="cookie-info">
                      <div className="cookie-name" style={{ fontSize: "22px" }}>
                        {cookieName}
                      </div>
                      <div className="predicted" style={{ fontSize: "20px" }}>
                        <strong>ML Predicted Cases (Historical):</strong>{" "}
                        {pred.predictedCases != null ? getAdjustedPredictedCases(pred.predictedCases).toFixed(1) : "--"}
                        <span style={{ fontSize: "16px", color: "#666", marginLeft: "10px" }}>
                          (100%: {pred.predictedCases != null ? get100PercentPredictedCases(pred.predictedCases).toFixed(1) : "--"})
                        </span>
                      </div>
                      
                      <div className="last-year-based" style={{ fontSize: "16px", color: "#666", marginTop: "8px" }}>
                        <strong>Last Year Sales Based (SIO):</strong>{" "}
                        {pred.last_year_based_prediction != null ? getAdjustedPredictedCases(pred.last_year_based_prediction).toFixed(1) : "No data"}
                        <span style={{ fontSize: "14px", color: "#999", marginLeft: "10px" }}>
                          (100%: {pred.last_year_based_prediction != null ? get100PercentPredictedCases(pred.last_year_based_prediction).toFixed(1) : "No data"})
                        </span>
                      </div>

                      {pred.predictedCases == null && (
                        <div className="no-data" style={{ fontSize: "14px", color: "#f0ad4e" }}>
                          No historical data available
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )
          )}
        </div>

        {/* Loading Spinner for analytics data */}
        <div className="analytics-wrapper">
          <div className="analytics-title">ANALYTICS</div>
          {loading ? (
            <div className="spinner" style={{ margin: '40px auto' }}></div>
          ) : error ? (
            <p style={{ color: "red" }}>{error}</p>
          ) : (
            girlsData.length > 0 && salesData.length > 0 && (
              <div className="analysis-section">
                <div className="analysis-box">
              <h4>Avg. Number of Girls by Year</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={girlsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" tickFormatter={periodToYear} />
                  <YAxis />
                  <Tooltip formatter={customTooltipFormatter} />
                  <Line type="monotone" dataKey="avgGirls" stroke="#8884d8" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            </div>
  
            <div className="analysis-box">
              <h4>Total Average Cases Sold by Year per Troop</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={salesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" tickFormatter={periodToYear} />
                  <YAxis />
                  <Tooltip formatter={customTooltipFormatter} />
                  <Bar dataKey="avgSales" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
  
            <div className="analysis-box" style={{ gridColumn: "span 2" }}>
              <h4>Girls vs. Cases Sold (Scatterplots for Each Cookie Type)</h4>
              <p>
                Each chart below shows data for one cookie type only. The X-axis is the Number of Girls and the Y-axis is Cases Sold.
              </p>
            </div>
  
            {cookieTypes.map((cookieType, idx) => {
              const filtered = scatterData.filter(
                (d) => d.canonical_cookie_type === cookieType
              );
              const trendlineData = getTrendline(filtered, "number_of_girls", "number_cases_sold");
              return (
                <div className="analysis-box" key={cookieType}>
                  <h5>{cookieType}</h5>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="number_of_girls" xAxisId="main" domain={["auto", "auto"]} />
                      <YAxis type="number" dataKey="number_cases_sold" yAxisId="main" domain={["auto", "auto"]} />
                      <Tooltip formatter={customTooltipFormatter} />
                      <Scatter data={filtered} fill={getColor(idx)} xAxisId="main" yAxisId="main" />
                      <Line
                        data={trendlineData}
                        dataKey="number_cases_sold"
                        xAxisId="main"
                        yAxisId="main"
                        stroke="#ff0000"
                      />
                      {predictedGirls && (
                        <ReferenceLine
                          xAxisId="main"
                          yAxisId="main"
                          x={Number(predictedGirls)}
                          stroke="#ffa500"
                          strokeWidth={4}
                          strokeDasharray="4 2"
                          label={{
                            value: `${predictedGirls} girls`,
                            position: "top",
                            fill: "#ffa500",
                            fontSize: 12,
                            fontWeight: "bold"
                          }}
                        />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              );
            })}
              </div>
            )
          )}
        </div>
      </div>
    );
  }
  

/* ------------------------------------------------------------------
   PAGE 2: ReturningTroopSearchPage
   - User enters troop id and number of girls.
   - Provides suggestions for troop id (fetched from /api/troop_ids).
   ------------------------------------------------------------------ */
   function ReturningTroopSearchPage({ onSearch, onBack, onAbout, onFaq, onHome }) {
    const [troopInput, setTroopInput] = useState("");
    const [numGirls, setNumGirls] = useState("");
    const [error, setError] = useState("");
    const [troopIds, setTroopIds] = useState([]);
  
    // Fetch all troop ids on mount
    useEffect(() => {
      const fetchTroopIds = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/troop_ids`);
          const data = await res.json();
          setTroopIds(data);
        } catch (err) {
          console.error("Error fetching troop ids:", err);
        }
      };
      fetchTroopIds();
    }, []);
  
      const handleSearch = () => {
    if (!troopInput || troopInput.trim().length === 0) {
      setError("Please enter a valid Troop ID.");
      return;
    }
    if (!numGirls || isNaN(numGirls)) {
      setError("Please enter a valid number of girls.");
      return;
    }
    const validatedGirls = Math.max(0, Math.min(250, Number(numGirls)));
    setError("");
    // Format troop ID as 5-character string with leading zeros for numerical values
    const troopId = troopInput.trim().match(/^\d+$/) ? troopInput.trim().padStart(5, '0') : troopInput.trim().padStart(5);
    onSearch(troopId, validatedGirls);
  };
  
    return (
      <div className="main-container">
        <div className="background"></div>
        <div className="overlay"></div>
        <header className="header">
          <div className="logo-row">
            <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
          </div>
          <nav className="nav-links">
            <div className="nav-link" onClick={onHome}><FaHome /></div>
            <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
            <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
          </nav>
        </header>
        <h1 className="title">Returning Troops</h1>
        <p className="subtitle">Enter your Troop ID (5 digits, e.g. 00001) and Number of Girls</p>
        <div className="input-container">
          <div className="input-box">
            Troop ID:{" "}
            <CustomDropdown
              value={troopInput}
              onChange={setTroopInput}
              options={troopIds}
              placeholder="e.g. 00001"
            />
          </div>
          <div className="input-box">
            Number of Girls:{" "}
            <input
              type="number"
              max="250"
              value={numGirls}
              onChange={(e) => setNumGirls(e.target.value)}
              placeholder="e.g. 25"
            />
          </div>
          <div style={{ display: "flex", gap: "20px", marginTop: "20px" }}>
            <button className="predict-button" onClick={onBack}>Back</button>
            <button className="predict-button" onClick={handleSearch}>Search</button>
          </div>
        </div>
        {/* Suggestions handled via datalist; legacy list removed */}
        {error && <p style={{ color: "red" }}>{error}</p>}
      </div>
    );
  }

/* ------------------------------------------------------------------
   PAGE 3: ReturningTroopAnalyticsPage
   - Displays analytics and predictions.
   - Also shows Troop ID and the associated SU info (returned from /api/history).
   ------------------------------------------------------------------ */
      function ReturningTroopAnalyticsPage({ troopId, numGirls, onBack, onAbout, onFaq, onHome }) {
    const [girlsData, setGirlsData] = useState([]);
    const [salesData, setSalesData] = useState([]);
    const [cookieBreakdownData, setCookieBreakdownData] = useState([]);
    const [predictions, setPredictions] = useState({});
    const [cookieTypes, setCookieTypes] = useState([]);
    const [loading, setLoading] = useState(false); // for analytics/history
    const [loadingPredictions, setLoadingPredictions] = useState(false); // for predictions only
    const [error, setError] = useState("");
    // Associated Service Unit info (returned by /api/history)
    const [suInfo, setSuInfo] = useState({ su: null, suName: null });
    // State for updating number of girls (prefilled with initial value)
    const [updatedNumGirls, setUpdatedNumGirls] = useState(Math.max(0, Math.min(250, numGirls)));
    // State for percentage field
    const [percentage, setPercentage] = useState("50");
    const [percentageError, setPercentageError] = useState("");
    const [appliedPercentage, setAppliedPercentage] = useState(50); // State for the percentage applied to predictions
  

  
    // Fetch history and cookie breakdown data when troopId changes
    useEffect(() => {
      if (!troopId) return;
      setLoading(true);
      setError("");
      const fetchHistory = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/history/${troopId}`);
          const history = await res.json();
          if (history.error) {
            setError(history.error);
          } else {
            setGirlsData(history.girlsByPeriod || []);
            setSalesData(history.totalSalesByPeriod || []);
            setSuInfo({ su: history.su, suName: history.suName });
          }
        } catch (err) {
          console.error("Error fetching history:", err);
          setError("Could not fetch history.");
        } finally {
          setLoading(false);
        }
      };
  
      const fetchBreakdown = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/cookie_breakdown/${troopId}`);
          const dataBreak = await res.json();
          setCookieBreakdownData(dataBreak);
          if (dataBreak.length > 0) {
            const keys = Object.keys(dataBreak[0]).filter((k) => k !== "period");
            setCookieTypes(keys);
          } else {
            setCookieTypes([]);
          }
        } catch (err) {
          console.error("Error fetching cookie breakdown:", err);
        }
      };
  
      fetchHistory();
      fetchBreakdown();
    }, [troopId]);
  
    // Function to fetch predictions using a given number of girls
    const fetchPredictions = async (girlsVal) => {
      setLoadingPredictions(true); // <-- Show spinner while fetching predictions
      try {

        const res = await fetch(`${API_BASE}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            troop_id: troopId,
            num_girls: Number(girlsVal),
            year: 2024,
          }),
        });
        const dataPred = await res.json();
        const formatted = {};
        dataPred.forEach((d) => {
          // Use the cookie_type as is, since backend now provides canonical names
          const key = d.cookie_type;
          formatted[key] = {
            predictedCases: d.predicted_cases,
            last_year_based_prediction: d.last_year_based_prediction,
            imageUrl: d.image_url, // backend provides correct URL
          };
        });
        setPredictions(formatted);
      } catch (err) {
        console.error("Error fetching predictions:", err);
      } finally {
        setLoadingPredictions(false); // <-- Hide spinner after fetching
      }
    };
  
    // Initial predictions fetch when troopId is set
    useEffect(() => {
      if (troopId) {
        if (Number(updatedNumGirls) <= 250) {
          fetchPredictions(updatedNumGirls);
        } else {
          alert("Number of girls cannot exceed 250.");
        }
      }
    }, [troopId]);
  
    const handleUpdatePredictions = () => {
      const clamped = Math.max(0, Math.min(250, Number(updatedNumGirls)));
      setUpdatedNumGirls(clamped);
      fetchPredictions(clamped);
      
      // Apply the percentage when button is clicked
      if (percentage && !isNaN(parseFloat(percentage))) {
        const percentageValue = parseFloat(percentage);
        if (percentageValue >= 0 && percentageValue <= 100) {
          setAppliedPercentage(percentageValue);
        }
      } else {
        setAppliedPercentage(100); // Default to 100% if no valid percentage
      }
    };
  
    const handlePercentageChange = (e) => {
      const value = e.target.value;
      setPercentage(value);
      
      // Clear error when user starts typing
      if (percentageError) {
        setPercentageError("");
      }
      
      // Validate percentage range
      const numValue = parseFloat(value);
      if (value !== "" && (isNaN(numValue) || numValue < 0 || numValue > 100)) {
        setPercentageError("Please enter a value between 0-100");
      } else {
        setPercentageError("");
      }
    };
  
        // Function to calculate adjusted predicted cases based on percentage
    const getAdjustedPredictedCases = (originalCases) => {
      if (!appliedPercentage || isNaN(appliedPercentage)) {
        return originalCases;
      }
      
      if (appliedPercentage < 0 || appliedPercentage > 100) {
        return originalCases;
      }
      
      // Convert percentage to multiplier (e.g., 50% = 0.5, 150% = 1.5)
      const multiplier = appliedPercentage / 100;
      return originalCases * multiplier;
    };

    // Function to calculate 100% predicted cases (for comparison)
    const get100PercentPredictedCases = (originalCases) => {
      return originalCases;
    };

    return (
      <div className="main-container">
        <div className="background"></div>
        <div className="overlay"></div>
        <header className="header">
          <div className="logo-row">
            <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
            <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
          </div>
          <nav className="nav-links">
            <div className="nav-link" onClick={onHome}><FaHome /></div>
            <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
            <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
          </nav>
        </header>
        <h1 className="title">Returning Troop Dashboard</h1>
        <p className="subtitle">
          Troop ID: {troopId}
          {suInfo.su && (
            <> (Service Unit #{suInfo.su} – {suInfo.suName})</>
          )}
        </p>
  
        {/* Input to update the number of girls */}
        <div className="input-container" style={{ marginBottom: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "20px", marginBottom: "10px" }}>
            <div style={{ display: "flex", alignItems: "center" }}>
              <label style={{ marginRight: "10px" }}>Update Number of Girls:</label>
              <input
                type="number"
                value={updatedNumGirls}
                onChange={(e) => setUpdatedNumGirls(e.target.value)}
                style={{ width: "80px" }}
                max="250"
                min="0"
              />
            </div>
            <div style={{ display: "flex", alignItems: "center" }}>
              <label style={{ marginRight: "10px" }}>Percentage (%):</label>
              <input
                type="number"
                value={percentage}
                onChange={handlePercentageChange}
                style={{ width: "80px" }}
                placeholder="0-100"
                step="0.1"
              />
            </div>
          </div>
          {percentageError && (
            <div style={{ color: "red", fontSize: "14px", marginBottom: "10px" }}>
              {percentageError}
            </div>
          )}
          <div style={{ display: "flex", gap: "20px" }}>
            <button className="predict-button" onClick={handleUpdatePredictions}>
              Update Predictions
            </button>
            <button className="predict-button" onClick={onBack}>Back</button>
          </div>
        </div>
  
        {/* Predictions Section */}
        <div style={{ fontSize: "18px", marginBottom: "30px" }}>
          <div className="predictions">PREDICTIONS</div>
          {loadingPredictions ? (
            <div className="spinner" style={{ margin: '40px auto' }}></div>
          ) : (
            <div className="cookie-grid" style={{ background: "none", padding: "20px" }}>
              {Object.entries(predictions).map(([cookieName, pred]) => (
                <div key={cookieName} className="cookie-box" style={{ fontSize: "18px" }}>
                  <img src={pred.imageUrl} alt={cookieName} />
                  <div className="cookie-info">
                    <div className="cookie-name" style={{ fontSize: "22px" }}>
                      {cookieName}
                    </div>
                    <div className="predicted" style={{ fontSize: "20px" }}>
                      <strong>ML Predicted Cases (Historical):</strong>{" "}
                      <span>
                        {pred.predictedCases != null ? getAdjustedPredictedCases(pred.predictedCases).toFixed(1) : "--"}
                      </span>
                      <span style={{ fontSize: "16px", color: "#666", marginLeft: "10px" }}>
                        (100%: {pred.predictedCases != null ? get100PercentPredictedCases(pred.predictedCases).toFixed(1) : "--"})
                      </span>
                    </div>
                    
                    <div className="last-year-based" style={{ fontSize: "16px", color: "#666", marginTop: "8px" }}>
                      <strong>Last Year Sales Based (SIO):</strong>{" "}
                      <span>
                        {pred.last_year_based_prediction != null ? getAdjustedPredictedCases(pred.last_year_based_prediction).toFixed(1) : "No data"}
                      </span>
                      <span style={{ fontSize: "14px", color: "#999", marginLeft: "10px" }}>
                        (100%: {pred.last_year_based_prediction != null ? get100PercentPredictedCases(pred.last_year_based_prediction).toFixed(1) : "No data"})
                      </span>
                    </div>

                    {pred.predictedCases == null && (
                      <div className="no-data" style={{ fontSize: "14px", color: "#f0ad4e" }}>
                        No historical data available
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Analytics Section */}
        <div className="analytics-title">ANALYTICS</div>
        <div className="analysis-section">
          
          {girlsData.length === 0 && <div style={{color: 'red'}}>No girlsData</div>}
          {salesData.length === 0 && <div style={{color: 'red'}}>No salesData</div>}
          {cookieBreakdownData.length === 0 && <div style={{color: 'red'}}>No cookieBreakdownData</div>}
          {cookieTypes.length === 0 && <div style={{color: 'red'}}>No cookieTypes</div>}
          {/* Total Cookie Cases Sold Chart */}
          <div className="analysis-box">
            <h4>Total Cookie Cases Sold</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={salesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tickFormatter={periodToYear} />
                <YAxis />
                <Tooltip formatter={customTooltipFormatter} />
                <Line type="monotone" dataKey="totalSales" stroke="#8884d8" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {/* Number of Girls by Year Chart */}
          <div className="analysis-box">
            <h4>Number of Girls by Year</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={girlsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tickFormatter={periodToYear} />
                <YAxis />
                <Tooltip formatter={customTooltipFormatter} />
                <Bar dataKey="numberOfGirls" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {/* Header for Cookie Type Line Charts */}
          <div className="analysis-box" style={{ gridColumn: "span 2" }}>
            <h4>Historical Cases Sold (Line Charts for Each Cookie Type)</h4>
            <p>
              Each chart below shows historical data for one cookie type only. 
            </p>
          </div>
          {/* Line Charts per Cookie Type */}
          {cookieTypes.map((ct, idx) => {
            const lineData = cookieBreakdownData.map((row) => ({
              period: row.period,
              sales: row[ct] || 0,
            }));
            return (
              <div className="analysis-box" key={ct}>
                <h4>{ct} Sales Over Time</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={lineData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="period" tickFormatter={periodToYear} />
                    <YAxis />
                    <Tooltip formatter={customTooltipFormatter} />
                    <Line type="monotone" dataKey="sales" stroke={getColor(idx)} strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      </div>
    );
  }
  
  
  
/* ------------------------------------------------------------------
   ABOUT PAGE
   ------------------------------------------------------------------ */
function AboutPage({ onBack, onHome, onFaq }) {
  return (
    <div className="main-container">
      <div className="background"></div>
      <div className="overlay"></div>
      <header className="header">
        <div className="logo-row">
          <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
          <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
          <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
        </div>
        <nav className="nav-links">
          <div className="nav-link" onClick={onHome}><FaHome /></div>
          <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
          <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
        </nav>
      </header>
      <h1 className="title">About The Platform</h1>
      <div className="content" style={{ 
        maxWidth: '1200px', 
        margin: '0 auto', 
        textAlign: 'left',
        color: 'white',
        padding: '20px'
      }}>
        
        <section style={{ 
          marginBottom: '60px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '40px',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <h2 style={{ 
            color: 'white', 
            borderBottom: '3px solid #8B5CF6', 
            paddingBottom: '15px', 
            marginBottom: '30px',
            fontSize: '28px',
            fontWeight: 'bold',
            textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
          }}>
            Project
          </h2>
          <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
            <div style={{ flex: '1' }}>
              <p style={{ lineHeight: '1.8', fontSize: '18px', marginBottom: '20px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                Advanced machine learning platform that transforms cookie sales forecasting.
              </p>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>ML Models:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>Ridge Regression with clustering algorithms</li>
                  <li>Dynamic model selection for optimal accuracy</li>
                  <li>Hybrid system combining multiple prediction methods</li>
                </ul>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>Results:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>1.35 cases improvement per troop per cookie type</li>
                  <li>181,000+ additional boxes sold across 1,400+ troops</li>
                  <li>Real-time predictions with accurate forecasting</li>
                </ul>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>Features:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>Scenario modeling for different participation levels</li>
                  <li>Comprehensive analytics dashboards</li>
                  <li>Data-driven inventory and marketing decisions</li>
                </ul>
              </div>
            </div>
            <div style={{ flex: '0 0 400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <img 
                src={`${API_BASE}/static/chemicalgsci.jpg`} 
                alt="Machine Learning Project" 
                style={{ 
                  width: '100%', 
                  height: '350px', 
                  objectFit: 'cover',
                  borderRadius: '8px',
                  boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)'
                }}
              />
            </div>
          </div>
        </section>

        <section style={{ 
          marginBottom: '60px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '40px',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <h2 style={{ 
            color: 'white', 
            borderBottom: '3px solid #8B5CF6', 
            paddingBottom: '15px', 
            marginBottom: '30px',
            fontSize: '28px',
            fontWeight: 'bold',
            textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
          }}>
            Girl Scouts of Central Indiana
          </h2>
          <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
            <div style={{ flex: '0 0 400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <img 
                src={`${API_BASE}/static/gsci_whitelogo.jpg`} 
                alt="Girl Scouts of Central Indiana" 
                style={{ 
                  width: '100%', 
                  height: '350px', 
                  objectFit: 'cover',
                  borderRadius: '8px',
                  boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)'
                }}
              />
            </div>
            <div style={{ flex: '1' }}>
              <p style={{ lineHeight: '1.8', fontSize: '18px', marginBottom: '20px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                Empowering girls across the region to develop leadership skills and make a positive impact.
              </p>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>Cookie Program Benefits:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>Business skills: goal setting, money management, customer service</li>
                  <li>Entrepreneurship and marketing strategy development</li>
                  <li>SMART goal setting and progress tracking</li>
                </ul>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>Educational Impact:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>Funds troop activities and community service projects</li>
                  <li>Supports camping trips, STEM workshops, and badge activities</li>
                  <li>Builds character and life skills through hands-on learning</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section style={{ 
          marginBottom: '60px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '40px',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <h2 style={{ 
            color: 'white', 
            borderBottom: '3px solid #8B5CF6', 
            paddingBottom: '15px', 
            marginBottom: '30px',
            fontSize: '28px',
            fontWeight: 'bold',
            textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
          }}>
            Krenicki Center for Business Analytics & Machine Learning
          </h2>
          <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
            <div style={{ flex: '1' }}>
              <p style={{ lineHeight: '1.8', fontSize: '18px', marginBottom: '20px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                Leading hub for business analytics and machine learning research at Purdue University.
              </p>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>Research Focus:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>Predictive analytics and optimization algorithms</li>
                  <li>Machine learning applications across healthcare, finance, retail</li>
                  <li>Data-driven approaches to complex business challenges</li>
                </ul>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>Industry Partnerships:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>Collaboration with Fortune 500 companies and startups</li>
                  <li>Real-world project experience for students</li>
                  <li>Innovative solutions driving business value and social impact</li>
                </ul>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#8B5CF6' }}>Student Programs:</strong>
                <ul style={{ margin: '10px 0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '16px', textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)' }}>
                  <li>Specialized courses in business analytics and data science</li>
                  <li>Capstone projects, hackathons, and research initiatives</li>
                  <li>Career preparation in rapidly evolving analytics field</li>
                </ul>
              </div>
            </div>
            <div style={{ flex: '0 0 400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <img 
                src={`${API_BASE}/static/gscicamp.png`} 
                alt="Krenicki Center" 
                style={{ 
                  width: '100%', 
                  height: '350px', 
                  objectFit: 'cover',
                  borderRadius: '8px',
                  boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)'
                }}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------
   FAQ PAGE
   ------------------------------------------------------------------ */
function FAQPage({ onBack, onHome, onAbout }) {
  const [openQuestion, setOpenQuestion] = useState(null);

  const faqData = [
    {
      question: "How does the cookie prediction system work?",
      answer: "Advanced machine learning models analyze historical sales data and troop participation to predict future cookie sales. The system automatically selects the most accurate prediction method for each troop-cookie combination."
    },
    {
      question: "What's the difference between 'New Troop' and 'Returning Troop' predictions?",
      answer: "Returning Troop predictions use the troop's own historical data. New Troop predictions use data from similar troops in the same Service Unit since no direct history exists."
    },
    {
      question: "How accurate are the predictions?",
      answer: "The model improves accuracy by 1.35 cases per troop per cookie type compared to traditional methods. Advanced machine learning algorithms provide accurate predictions for better planning."
    },
    {
      question: "Can I adjust predictions for different scenarios?",
      answer: "Yes! Modify the number of girls participating and apply percentage adjustments to see how different factors affect predicted sales and plan for various participation levels."
    },
    {
      question: "What should I do with these predictions?",
      answer: "Use predictions for inventory ordering, setting realistic sales goals, and planning fundraising strategy. Analytics help identify trends and opportunities to improve performance."
    }
  ];

  const toggleQuestion = (index) => {
    setOpenQuestion(openQuestion === index ? null : index);
  };

  return (
    <div className="main-container">
      <div className="background"></div>
      <div className="overlay"></div>
      <header className="header">
        <div className="logo-row">
          <img src={`${API_BASE}/static/GSC(2).png`} alt="GSCI Logo" style={{ height: '50px', marginRight: '20px' }} />
          <img src={`${API_BASE}/static/KREN2.png`} alt="KREN2 Logo" style={{ height: '100px', marginRight: '20px' }} />
          <img src={`${API_BASE}/static/KREN.png`} alt="KREN Logo" style={{ height: '150px' }} />
        </div>
        <nav className="nav-links">
          <div className="nav-link" onClick={onHome}><FaHome /></div>
          <div className="nav-link" onClick={onFaq}><FaQuestionCircle /></div>
          <div className="nav-link" onClick={onAbout}><FaInfoCircle /></div>
        </nav>
      </header>
      <h1 className="title">Frequently Asked Questions</h1>
      <div className="content" style={{ 
        maxWidth: '900px', 
        margin: '0 auto', 
        textAlign: 'left',
        color: 'white',
        padding: '20px'
      }}>
        
        <div style={{ 
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '40px',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          {faqData.map((faq, index) => (
            <div key={index} style={{ marginBottom: index < faqData.length - 1 ? '20px' : '0' }}>
              <button
                onClick={() => toggleQuestion(index)}
                style={{
                  width: '100%',
                  padding: '25px',
                  backgroundColor: openQuestion === index ? 'rgba(139, 92, 246, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                  color: 'white',
                  border: openQuestion === index ? '2px solid #8B5CF6' : '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '8px',
                  textAlign: 'left',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)'
                }}
              >
                {faq.question}
                <FaChevronDown 
                  style={{ 
                    transform: openQuestion === index ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.3s ease',
                    color: openQuestion === index ? '#8B5CF6' : 'white'
                  }} 
                />
              </button>
              
              {openQuestion === index && (
                <div style={{
                  padding: '25px',
                  backgroundColor: 'rgba(0, 0, 0, 0.3)',
                  border: '1px solid rgba(139, 92, 246, 0.3)',
                  borderTop: 'none',
                  borderRadius: '0 0 8px 8px',
                  fontSize: '16px',
                  lineHeight: '1.7',
                  color: 'white',
                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)',
                  marginTop: '2px'
                }}>
                  {faq.answer}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------
   MAIN APP COMPONENT: Manages navigation between pages.
   ------------------------------------------------------------------ */
   function App() {
    const [page, setPage] = useState("landing");
    
    // For New Troop flow (unchanged)
    const [selectedSU, setSelectedSU] = useState(null);
    const [selectedSUName, setSelectedSUName] = useState("");
    const [newTroopNumGirls, setNewTroopNumGirls] = useState("");
    // For Returning Troop flow:
    const [returningTroopId, setReturningTroopId] = useState("");
    const [returningNumGirls, setReturningNumGirls] = useState("");
  
    if (page === "landing") {
      return (
        <LandingPage
          onNewTroop={() => setPage("newTroopSearch")}
          onReturningTroop={() => setPage("returningTroopSearch")}
          onManual={() => setPage("manual")}
          onAbout={() => setPage("about")}
          onFaq={() => setPage("faq")}
          onHome={() => setPage("landing")}
        />
      );
    }
  
    if (page === "newTroopSearch") {
      return (
        <NewTroopSearchPage
          onBack={() => setPage("landing")}
          onSearch={(suNumber, suName, numGirls) => {
            setSelectedSU(suNumber);
            setSelectedSUName(suName);
            setNewTroopNumGirls(numGirls);
            setPage("newTroopAnalytics");
          }}
          onAbout={() => setPage("about")}
          onFaq={() => setPage("faq")}
          onHome={() => setPage("landing")}
        />
      );
    }

    if (page === "manual") {
      return <ManualPage onBack={() => setPage("landing")} onAbout={() => setPage("about")} onFaq={() => setPage("faq")} onHome={() => setPage("landing")} />;
    }
    
  
    if (page === "newTroopAnalytics") {
      return (
        <NewTroopAnalyticsPage
          suNumber={selectedSU}
          suName={selectedSUName}
          initialNumGirls={newTroopNumGirls}
          onBack={() => setPage("newTroopSearch")}
          onAbout={() => setPage("about")}
          onFaq={() => setPage("faq")}
          onHome={() => setPage("landing")}
        />
      );
    }
  
    if (page === "returningTroopSearch") {
      return (
        <ReturningTroopSearchPage
          onBack={() => setPage("landing")}
          onSearch={(troopId, numGirls) => {
            setReturningTroopId(troopId);
            setReturningNumGirls(numGirls);
            setPage("returningTroopAnalytics");
          }}
          onAbout={() => setPage("about")}
          onFaq={() => setPage("faq")}
          onHome={() => setPage("landing")}
        />
      );
    }
  
    if (page === "returningTroopAnalytics") {
      return (
        <ReturningTroopAnalyticsPage
          troopId={returningTroopId}
          numGirls={returningNumGirls}
          onBack={() => setPage("returningTroopSearch")}
          onAbout={() => setPage("about")}
          onFaq={() => setPage("faq")}
          onHome={() => setPage("landing")}
        />
      );
    }

    if (page === "about") {
      return <AboutPage onBack={() => setPage("landing")} onHome={() => setPage("landing")} onFaq={() => setPage("faq")} />;
    }

    if (page === "faq") {
      return <FAQPage onBack={() => setPage("landing")} onHome={() => setPage("landing")} onAbout={() => setPage("about")} />;
    }
  
    return null;
  }
  
  export default App;
