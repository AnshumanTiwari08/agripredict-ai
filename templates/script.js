document.addEventListener('DOMContentLoaded', () => {
  // Initialize AOS
  AOS.init({
    duration: 1000, // Animation duration
    once: true,      // Whether animation should happen only once - while scrolling down
    mirror: false,   // Whether elements should animate out while scrolling past them
  });

  // Register GSAP plugins
  gsap.registerPlugin(ScrollTrigger);

  // Highlight active navigation link
  const currentPath = window.location.pathname.split('/').pop(); // Get just the filename
  const navLinks = document.querySelectorAll('nav ul li a');

  navLinks.forEach(link => {
    const linkPath = link.getAttribute('href').split('/').pop();
    if (linkPath === currentPath || (currentPath === '' && linkPath === 'index.html')) {
      link.classList.add('active');
    }
  });

  // Smooth scroll for anchor links (if any, though navbar uses full page links)
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      document.querySelector(this.getAttribute('href')).scrollIntoView({
        behavior: 'smooth'
      });
    });
  });

  // 3D tilt effect for containers (if applicable, e.g., on predict.html)
  // This is a CSS-only hover effect now, but GSAP could enhance it further if needed.
  // The CSS handles the hover transform, JS handles AOS and GSAP.

  // --- GSAP Scroll Animations ---

  // Parallax for decorative elements (leaves/grains)
  gsap.utils.toArray(".decorative-element").forEach(element => {
    gsap.to(element, {
      yPercent: () => gsap.utils.random(20, 50), // Move 20-50% of its height relative to scroll
      ease: "none",
      scrollTrigger: {
        trigger: "body",
        start: "top top",
        end: "bottom top",
        scrub: 1, // Smoothly animate on scroll
      }
    });
  });

  // Parallax for floating circles (move at a different speed)
  gsap.utils.toArray(".floating-circle").forEach(circle => {
    gsap.to(circle, {
      yPercent: () => gsap.utils.random(-10, -30), // Move -10 to -30% of its height relative to scroll
      xPercent: () => gsap.utils.random(-5, 5),
      ease: "none",
      scrollTrigger: {
        trigger: "body",
        start: "top top",
        end: "bottom top",
        scrub: 0.8, // Slightly faster scrub
      }
    });
  });

  // Example: Hero section text animation on scroll (index.html specific)
  const heroH1 = document.querySelector('.hero-section h1');
  const heroP = document.querySelector('.hero-section p');
  const heroButton = document.querySelector('.hero-section .button');

  if (heroH1 && heroP && heroButton) {
    gsap.from([heroH1, heroP, heroButton], {
      y: 50,
      opacity: 0,
      stagger: 0.2,
      duration: 1,
      ease: "power3.out",
      scrollTrigger: {
        trigger: ".hero-section",
        start: "top 80%", // When the top of the hero section hits 80% of viewport
        toggleActions: "play none none reverse", // Play on enter, reverse on leave back up
      }
    });
  }

  // Example: Container zoom/fade on scroll (for all .container elements)
  gsap.utils.toArray(".container").forEach(container => {
    // AOS already handles initial fade-in, this adds a subtle continuous effect
    gsap.to(container, {
      scale: 0.98, // Slightly shrink as you scroll past
      opacity: 0.8, // Fade out slightly
      ease: "power1.in",
      scrollTrigger: {
        trigger: container,
        start: "top center", // Start when top of container hits center of viewport
        end: "bottom top", // End when bottom of container leaves top of viewport
        scrub: 0.5,
        // markers: true, // Uncomment for debugging scrolltrigger
      }
    });
  });

  // Example: Timeline items (about.html) - staggered slide in
  const timelineItems = document.querySelectorAll('.timeline-item');
  if (timelineItems.length > 0) {
    gsap.from(timelineItems, {
      x: -100,
      opacity: 0,
      stagger: 0.2,
      duration: 0.8,
      ease: "power2.out",
      scrollTrigger: {
        trigger: ".container", // Trigger when the main container for timeline items enters
        start: "top 70%",
        toggleActions: "play none none reverse",
      }
    });
  }

  // Example: India Map (eda.html) - zoom in on scroll
  const indiaMap = document.querySelector('.india-map-container svg');
  if (indiaMap) {
    gsap.from(indiaMap, {
      scale: 0.8,
      opacity: 0,
      duration: 1,
      ease: "back.out(1.7)",
      scrollTrigger: {
        trigger: ".india-map-container",
        start: "top 80%",
        toggleActions: "play none none reverse",
      }
    });
  }

  // Example: Chart containers (eda.html) - staggered fade in from sides
  const chartContainers = document.querySelectorAll('.chart-container');
  if (chartContainers.length > 0) {
    gsap.from(chartContainers, {
      x: (i, target) => i % 2 === 0 ? -100 : 100, // Alternate slide from left/right
      opacity: 0,
      stagger: 0.3,
      duration: 0.8,
      ease: "power2.out",
      scrollTrigger: {
        trigger: chartContainers[0].closest('.container'), // Trigger when parent container enters
        start: "top 75%",
        toggleActions: "play none none reverse",
      }
    });
  }

});
