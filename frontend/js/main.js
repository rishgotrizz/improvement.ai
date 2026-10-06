/**
 * EDUPREDICT - Shared Main JavaScript & API Service Layer
 * First-Year Engineering & AI Academic Project
 * Phase 2: Functional SQLite Data API Integration
 */

// Centralized Mock/Demo Data Store (Fallback when API is offline)
const EduPredictData = {
  demoStudent: {
    studentId: "STU-2026-001",
    name: "Demo Student (First Year)",
    semester: 2,
    branch: "Computer Engineering",
    academicScore: 78.4,
    attendance: 87.0,           // Attendance percentage (0-100%)
    studyHours: 18.0,           // Weekly self-study hours
    previousPercentage: 79.0,   // Previous semester percentage (0-100%)
    assignmentAvg: 82.5,        // Assignment average score (%)
    internalAssessment: 76.0,   // Internal assessment score (%)
    quizAvg: 80.0,              // Quiz average percentage (%)
    completedAssignments: 14,   // Completed coursework count
    totalAssignments: 15,
    backlogCount: 0,            // Active failed/uncleared subjects
    riskCategory: "PENDING",    // Model training pending
    riskProbability: 0.142,
    
    // Student Academic Performance Subjects (Actual curriculum courses)
    studentSubjects: [
      { code: "CS102", name: "Data Structures", score: 74.0, attendance: 80.0, assignments: 75.0, priority: "HIGH" },
      { code: "MATH101", name: "Mathematics", score: 72.0, attendance: 85.0, assignments: 78.0, priority: "MEDIUM" },
      { code: "PHY101", name: "Physics", score: 81.0, attendance: 88.0, assignments: 85.0, priority: "LOW" },
      { code: "CS101", name: "Computer Programming", score: 88.0, attendance: 92.0, assignments: 90.0, priority: "LOW" },
      { code: "EE101", name: "Basic Electrical Engineering", score: 84.0, attendance: 90.0, assignments: 88.0, priority: "LOW" }
    ]
  },
  
  // Project Academic Subjects (The 6 First-Year Subjects mapping to EduPredict features)
  projectSubjects: [
    {
      code: "AM-I",
      name: "Applied Mathematics-I",
      icon: "📐",
      colorClass: "am",
      topics: ["Descriptive Statistics", "Linear Regression", "MSE Loss", "Pearson Correlation", "Gradient Descent"],
      role: "Provides the mathematical and statistical foundation for regression modeling, loss functions, and dataset metric analysis."
    },
    {
      code: "CPA",
      name: "Computer Programming & Algorithms",
      icon: "💻",
      colorClass: "cpa",
      topics: ["Input Validation", "Sorting Algorithms", "Subject Priority Logic", "Rule-Engine Categorization"],
      role: "Powers structured input validation routines, data processing logic, and subject study time prioritization algorithms."
    },
    {
      code: "COA",
      name: "Computer Organization & Architecture",
      icon: "⚙️",
      colorClass: "coa",
      topics: ["Time Complexity", "Space Complexity", "Memory-Aware Data Handling", "Single-Pass Dataset Aggregation"],
      role: "Enforces memory-efficient data handling, minimal redundant computing, and dataset parsing optimization."
    },
    {
      code: "IAI",
      name: "Introduction to Artificial Intelligence",
      icon: "🤖",
      colorClass: "iai",
      topics: ["Supervised Learning", "Logistic Classification", "Score Regression", "Evaluation Metrics (F1, MAE)"],
      role: "Implements the core machine learning models classifying academic risk levels and predicting semester percentage."
    },
    {
      code: "EDA",
      name: "Exploratory Data Analysis",
      icon: "📊",
      colorClass: "eda",
      topics: ["Data Cleaning", "Feature Correlation", "Distributions & Outliers", "Pandas & Matplotlib Visuals"],
      role: "Drives the analytics dashboard, discovering statistical relationships and correlation heatmaps across student attributes."
    },
    {
      code: "EAI",
      name: "Ethics in Artificial Intelligence",
      icon: "⚖️",
      colorClass: "eai",
      topics: ["Data Privacy", "Model Explainability", "Demographic Fairness Disclaimer", "Human-in-the-Loop Safeguards"],
      role: "Ensures responsible AI governance: data minimization, transparent feature contribution weights, and human oversight."
    }
  ]
};

/**
 * EduPredict API Service Layer
 * Centralized async methods for backend communication.
 */
const EduPredictAPI = {
  // Fetch Student Profile from SQLite API
  async getStudent(studentId = "STU-2026-001") {
    try {
      const response = await fetch(`/api/student/${studentId}`);
      if (response.ok) return await response.json();
    } catch (e) {
      console.warn("[EduPredictAPI] Student API unavailable; using fallback demo store.", e);
    }
    return EduPredictData.demoStudent;
  },

  // Fetch Student Assessment History
  async getStudentHistory(studentId = "STU-2026-001") {
    try {
      const response = await fetch(`/api/student/${studentId}/history`);
      if (response.ok) return await response.json();
    } catch (e) {
      console.warn("[EduPredictAPI] History API unavailable.", e);
    }
    return { studentId, count: 1, history: [EduPredictData.demoStudent] };
  },

  // Fetch Course Performance Breakdowns
  async getStudentSubjects(studentId = "STU-2026-001") {
    try {
      const response = await fetch(`/api/student/${studentId}/subjects`);
      if (response.ok) {
        const res = await response.json();
        return res.subjects || [];
      }
    } catch (e) {
      console.warn("[EduPredictAPI] Subjects API unavailable.", e);
    }
    return EduPredictData.demoStudent.studentSubjects;
  },

  // Post Assessment Submission to Backend API
  async submitAssessment(assessmentPayload) {
    try {
      const response = await fetch("/api/assessment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(assessmentPayload)
      });
      const data = await response.json();
      return { status: response.status, data };
    } catch (e) {
      console.error("[EduPredictAPI] Network/Server Error submitting assessment:", e);
      return {
        status: 500,
        data: {
          success: false,
          errors: { server: "Unable to connect to backend server. Please verify Flask startup." }
        }
      };
    }
  },

  // Fetch Assessment Record by ID
  async getAssessment(assessmentId) {
    try {
      const response = await fetch(`/api/assessment/${assessmentId}`);
      if (response.ok) return await response.json();
    } catch (e) {}
    return null;
  },

  // Fetch Dataset Quality Report
  async getDataQuality() {
    try {
      const response = await fetch("/api/data/quality");
      if (response.ok) return await response.json();
    } catch (e) {}
    return { totalRecords: 500, totalMissingValues: 0, status: "HEALTHY", isDemoDataset: true };
  },

  // Fetch AI Model Evaluation Metrics (Phase 4)
  async getModelMetrics() {
    try {
      const response = await fetch("/api/model/metrics");
      if (response.ok) return await response.json();
    } catch (e) {
      console.warn("[EduPredictAPI] Model metrics API unavailable.", e);
    }
    return null;
  },

  // Run AI Machine Learning Risk Classification & Score Prediction
  async predictRisk(payload) {
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      return { status: response.status, data };
    } catch (e) {
      console.error("[EduPredictAPI] Network/Server Error making ML prediction:", e);
      return {
        status: 500,
        data: { success: false, error: "Network error connecting to ML prediction service." }
      };
    }
  }
};

// UI DOM Lifecycle Setup
document.addEventListener("DOMContentLoaded", () => {
  highlightActiveNavLink();
  setupMobileNavigation();
  updateUserSessionUI();
  setupProtectedLinkInterception();
});

// Update navbar and role badges based on session state
async function updateUserSessionUI() {
  let user = null;
  const localRaw = localStorage.getItem("edupredict_user");
  if (localRaw) {
    try { user = JSON.parse(localRaw); } catch(e) {}
  }

  try {
    const res = await fetch("/api/auth/me");
    if (res.ok) {
      const data = await res.json();
      if (data.authenticated && data.user) {
        user = data.user;
        localStorage.setItem("edupredict_user", JSON.stringify(user));
      } else {
        localStorage.removeItem("edupredict_user");
      }
    }
  } catch(e) {}

  const navMenu = document.querySelector(".nav-menu");
  if (navMenu && user) {
    let facultyLink = navMenu.querySelector('a[href="/faculty"]');
    if (user.role === "faculty" && !facultyLink) {
      const linkHtml = `<a href="/faculty" class="nav-link">Faculty Dashboard</a>`;
      navMenu.insertAdjacentHTML("afterbegin", linkHtml);
      highlightActiveNavLink();
    }
  }
}

// Highlight current active route link in navbar
function highlightActiveNavLink() {
  const currentPath = window.location.pathname.replace(/\/$/, "") || "/";
  const navLinks = document.querySelectorAll(".nav-link");
  
  navLinks.forEach(link => {
    let href = link.getAttribute("href").replace(/\/$/, "");
    if (href === "") href = "/";
    
    if (href === currentPath || (currentPath === "/" && href === "/index.html")) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });
}

// Mobile Hamburger Drawer Listener
function setupMobileNavigation() {
  const mobileToggle = document.querySelector(".mobile-toggle");
  const navMenu = document.querySelector(".nav-menu");
  
  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener("click", () => {
      navMenu.classList.toggle("open");
    });
  }
}

// Global protected link click interception for unauthenticated users
function setupProtectedLinkInterception() {
  document.addEventListener("click", (e) => {
    const targetLink = e.target.closest("a[href='/prediction'], a[href='/xai'], a[href='/what-if']");
    if (!targetLink) return;

    // Call e.preventDefault() SYNCHRONOUSLY to stop instant browser navigation
    e.preventDefault();

    const href = targetLink.getAttribute("href");
    const targetUrl = href.includes("what-if") ? "/what-if" : "/prediction";

    (async () => {
      let isAuthed = false;
      try {
        const res = await fetch("/api/auth/me");
        if (res.ok) {
          const data = await res.json();
          isAuthed = Boolean(data && data.authenticated && data.user);
        }
      } catch(err) {}

      if (isAuthed) {
        window.location.href = targetUrl;
      } else {
        const loginModal = document.getElementById("login-required-modal");
        if (loginModal) {
          localStorage.setItem("edupredict_target_url", targetUrl);
          loginModal.classList.add("active");
        } else {
          window.location.href = "/login?next=" + encodeURIComponent(targetUrl);
        }
      }
    })();
  });
}
