import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import sympy as sp

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

st.title("📚 함수의 기본 개념")
st.markdown("---")

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["다항 함수", "삼각 함수", "지수 함수", "로그 함수", "초월 함수"]
)

# 1. 다항 함수
with tab1:
    st.subheader("다항 함수 (Polynomial Function)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("""
        ### 정의
        다항 함수는 다음과 같은 형태의 함수입니다:
        
        **f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀**
        
        여기서 n은 음이 아닌 정수이고, aᵢ는 상수입니다.
        
        ### 분류
        - **1차 함수**: f(x) = ax + b (직선)
        - **2차 함수**: f(x) = ax² + bx + c (포물선)
        - **3차 함수**: f(x) = ax³ + bx² + cx + d
        - **n차 함수**: n이 자연수인 경우
        
        ### 주요 특징
        - 실수 전체에서 연속
        - 정의역: 실수 전체
        - 치역: 함수의 차수와 최고차 계수의 부호에 따라 결정
        - 끝의 거동: 차수와 최고차 계수로 결정
        """)
    
    with col2:
        # 다항 함수 그래프들
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle('다항 함수의 예시', fontsize=14)
        
        x = np.linspace(-3, 3, 1000)
        
        # 1차 함수
        axes[0, 0].plot(x, 2*x + 1, 'b-', linewidth=2)
        axes[0, 0].set_title('1차 함수: f(x) = 2x + 1')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
        axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
        
        # 2차 함수
        axes[0, 1].plot(x, x**2 - 2, 'r-', linewidth=2)
        axes[0, 1].set_title('2차 함수: f(x) = x² - 2')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
        axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
        
        # 3차 함수
        axes[1, 0].plot(x, x**3 - 3*x, 'g-', linewidth=2)
        axes[1, 0].set_title('3차 함수: f(x) = x³ - 3x')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
        axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
        
        # 4차 함수
        axes[1, 1].plot(x, x**4 - 5*x**2 + 4, 'm-', linewidth=2)
        axes[1, 1].set_title('4차 함수: f(x) = x⁴ - 5x² + 4')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
        axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)


# 2. 삼각 함수
with tab2:
    st.subheader("삼각 함수 (Trigonometric Function)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("""
        ### 정의
        삼각 함수는 각도와 변의 비의 관계를 나타내는 함수입니다.
        
        ### 주요 삼각 함수
        - **사인 함수**: sin(x)
        - **코사인 함수**: cos(x)
        - **탄젠트 함수**: tan(x)
        - **코시컨트**: csc(x) = 1/sin(x)
        - **시컨트**: sec(x) = 1/cos(x)
        - **코탄젠트**: cot(x) = 1/tan(x)
        
        ### 주요 특징
        - **주기**: 
          - sin(x), cos(x): 2π
          - tan(x): π
        - **진폭**: sin(x), cos(x)는 [-1, 1]
        - **정의역**: 
          - sin(x), cos(x): 모든 실수
          - tan(x): x ≠ π/2 + nπ
        """)
    
    with col2:
        fig, axes = plt.subplots(3, 1, figsize=(8, 9))
        fig.suptitle('삼각 함수의 그래프', fontsize=14)
        
        x = np.linspace(-2*np.pi, 2*np.pi, 1000)
        
        # sin 함수
        axes[0].plot(x, np.sin(x), 'b-', linewidth=2)
        axes[0].set_title('sin(x)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1.5, 1.5)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].axvline(x=0, color='k', linewidth=0.5)
        
        # cos 함수
        axes[1].plot(x, np.cos(x), 'r-', linewidth=2)
        axes[1].set_title('cos(x)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-1.5, 1.5)
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        axes[1].axvline(x=0, color='k', linewidth=0.5)
        
        # tan 함수 (불연속점 제외)
        x_tan = np.linspace(-np.pi, np.pi, 1000)
        y_tan = np.tan(x_tan)
        axes[2].plot(x_tan, np.clip(y_tan, -5, 5), 'g-', linewidth=2)
        axes[2].set_title('tan(x) [범위 제한: -5 to 5]')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='k', linewidth=0.5)
        axes[2].axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)


# 3. 지수 함수
with tab3:
    st.subheader("지수 함수 (Exponential Function)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("""
        ### 정의
        지수 함수는 다음과 같은 형태의 함수입니다:
        
        **f(x) = aˣ** (a > 0, a ≠ 1)
        
        또는 더 일반적으로: **f(x) = b·aˣ⁺ᶜ + d**
        
        ### 주요 특징
        - **밑의 범위**: a > 0, a ≠ 1
        - **정의역**: 모든 실수
        - **치역**: (0, ∞) (항상 양수)
        - **y절편**: f(0) = 1 (기본 형태)
        - **점근선**: y = 0 (x축)
        - **연속성**: 모든 점에서 연속
        
        ### 성질
        - **a > 1**: 증가함수
        - **0 < a < 1**: 감소함수
        - **기본 밑**: e ≈ 2.71828 (자연지수)
        """)
    
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        fig.suptitle('지수 함수의 그래프', fontsize=14)
        
        x = np.linspace(-3, 3, 1000)
        
        # a > 1인 경우
        axes[0].plot(x, 2**x, 'b-', linewidth=2, label='f(x) = 2ˣ')
        axes[0].plot(x, np.e**x, 'r-', linewidth=2, label='f(x) = eˣ')
        axes[0].plot(x, 3**x, 'g-', linewidth=2, label='f(x) = 3ˣ')
        axes[0].set_title('a > 1 (증가)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1, 10)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].axvline(x=0, color='k', linewidth=0.5)
        axes[0].legend()
        
        # 0 < a < 1인 경우
        axes[1].plot(x, 0.5**x, 'b-', linewidth=2, label='f(x) = (1/2)ˣ')
        axes[1].plot(x, np.e**(-x), 'r-', linewidth=2, label='f(x) = e⁻ˣ')
        axes[1].plot(x, (1/3)**x, 'g-', linewidth=2, label='f(x) = (1/3)ˣ')
        axes[1].set_title('0 < a < 1 (감소)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-1, 10)
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        axes[1].axvline(x=0, color='k', linewidth=0.5)
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)


# 4. 로그 함수
with tab4:
    st.subheader("로그 함수 (Logarithmic Function)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("""
        ### 정의
        로그 함수는 지수 함수의 역함수입니다:
        
        **f(x) = log_a(x)** (a > 0, a ≠ 1)
        
        또는 더 일반적으로: **f(x) = b·log_a(x - c) + d**
        
        ### 주요 특징
        - **밑의 범위**: a > 0, a ≠ 1
        - **정의역**: (0, ∞) (양수만)
        - **치역**: 모든 실수
        - **x절편**: f(1) = 0
        - **점근선**: x = 0 (y축)
        - **연속성**: 정의역 내에서 연속
        
        ### 특별한 로그
        - **자연로그**: ln(x) = log_e(x)
        - **상용로그**: log(x) = log₁₀(x)
        """)
    
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        fig.suptitle('로그 함수의 그래프', fontsize=14)
        
        x = np.linspace(0.01, 10, 1000)
        
        # a > 1인 경우
        axes[0].plot(x, np.log2(x), 'b-', linewidth=2, label='f(x) = log₂(x)')
        axes[0].plot(x, np.log(x), 'r-', linewidth=2, label='f(x) = ln(x)')
        axes[0].plot(x, np.log10(x), 'g-', linewidth=2, label='f(x) = log₁₀(x)')
        axes[0].set_title('a > 1 (증가)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-3, 3)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].axvline(x=0, color='k', linewidth=0.5)
        axes[0].legend()
        axes[0].set_xlim(0, 10)
        
        # 0 < a < 1인 경우
        axes[1].plot(x, -np.log2(x), 'b-', linewidth=2, label='f(x) = -log₂(x)')
        axes[1].plot(x, -np.log(x), 'r-', linewidth=2, label='f(x) = -ln(x)')
        axes[1].plot(x, -np.log10(x), 'g-', linewidth=2, label='f(x) = -log₁₀(x)')
        axes[1].set_title('0 < a < 1 (감소)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-3, 3)
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        axes[1].axvline(x=0, color='k', linewidth=0.5)
        axes[1].legend()
        axes[1].set_xlim(0, 10)
        
        plt.tight_layout()
        st.pyplot(fig)


# 5. 초월 함수
with tab5:
    st.subheader("초월 함수 (Transcendental Function)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("""
        ### 정의
        초월 함수는 다항식으로 표현될 수 없는 함수입니다.
        위에서 학습한 삼각함수, 지수함수, 로그함수는 모두 초월함수입니다.
        
        ### 초월 함수의 종류
        - **삼각 함수**: sin, cos, tan 등
        - **역삼각 함수**: arcsin, arccos, arctan 등
        - **지수 함수**: aˣ (a > 0, a ≠ 1)
        - **로그 함수**: log_a(x)
        - **쌍곡함수**: sinh, cosh, tanh
        - **혼합형**: x²sin(x), e^x·cos(x) 등
        
        ### 주요 특징
        - 다항식으로는 표현 불가
        - 대부분 무한 급수로 표현 가능
        - 복잡한 변환 특성
        - 실생활에서 널리 사용됨 (파동, 성장, 감소 등)
        """)
    
    with col2:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle('초월 함수의 예시', fontsize=14)
        
        x = np.linspace(-2*np.pi, 2*np.pi, 1000)
        x_pos = np.linspace(0.01, 3*np.pi, 1000)
        
        # x*sin(x)
        axes[0, 0].plot(x, x*np.sin(x), 'b-', linewidth=2)
        axes[0, 0].set_title('f(x) = x·sin(x)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
        axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
        
        # e^x * cos(x)
        x_exp = np.linspace(-2, 2, 1000)
        axes[0, 1].plot(x_exp, np.e**x_exp*np.cos(x_exp), 'r-', linewidth=2)
        axes[0, 1].set_title('f(x) = eˣ·cos(x)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
        axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
        
        # sinh(x)
        axes[1, 0].plot(x_exp, np.sinh(x_exp), 'g-', linewidth=2)
        axes[1, 0].set_title('f(x) = sinh(x) = (eˣ - e⁻ˣ)/2')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
        axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
        
        # sin(x)/x (sinc 함수)
        axes[1, 1].plot(x, np.sinc(x/np.pi), 'm-', linewidth=2)
        axes[1, 1].set_title('f(x) = sin(x)/x')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
        axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.info("💡 다음 페이지의 함수 분석 도구에서 이런 함수들의 특성을 직접 분석해볼 수 있습니다!")
