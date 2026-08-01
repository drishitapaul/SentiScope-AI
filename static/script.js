document.addEventListener("DOMContentLoaded", () => {

    // ==========================
    // Animate Progress Bar
    // ==========================
    const progress = document.querySelector(".progress-fill");

    if (progress) {

        const targetWidth = progress.style.width;

        progress.style.width = "0%";

        setTimeout(() => {
            progress.style.width = targetWidth;
        }, 300);

    }

    // ==========================
    // Button Loading Animation
    // ==========================
    const form = document.querySelector("form");
    const button = document.querySelector(".analyze-btn");

    if (form && button) {

        form.addEventListener("submit", () => {

            button.disabled = true;

            button.innerHTML = `
                <span class="loader"></span>
                Analyzing...
            `;

        });

    }

    // ==========================
    // Fade-in Result Card
    // ==========================
    const result = document.querySelector(".result-card");

    if (result) {

        result.style.opacity = "0";
        result.style.transform = "translateY(20px)";

        setTimeout(() => {

            result.style.transition = "all .8s ease";

            result.style.opacity = "1";
            result.style.transform = "translateY(0px)";

        }, 250);

    }

    // ==========================
    // Auto Focus Textarea
    // ==========================
    const textarea = document.querySelector("textarea");

    if (textarea) {

        textarea.focus();

    }
// ==========================
// About Modal
// ==========================

const aboutBtn = document.getElementById("aboutBtn");
const modal = document.getElementById("aboutModal");
const closeModal = document.getElementById("closeModal");

if (aboutBtn && modal && closeModal) {

    aboutBtn.addEventListener("click", () => {
        modal.style.display = "flex";
    });

    closeModal.addEventListener("click", () => {
        modal.style.display = "none";
    });

    window.addEventListener("click", (e) => {
        if (e.target === modal) {
            modal.style.display = "none";
        }
    });

}
});
