//! nam-daisy-rs firmware: real-time NAM WaveNet-A1 inference on a Daisy Seed.
//!
//! Target: **Daisy Seed Rev1.2 / Seed2-DFM (PCM3060, hardware mode)**.
//!
//! Audio: SAI 24-bit interleaved `u32` -> mono -> input gain + denormal dither
//! -> `nam_core_nostd::Engine` -> interleave. Diagnostics are streamed over a
//! USB CDC serial port (readable from the host) and the onboard LED heartbeats.

#![no_std]
#![no_main]

use core::fmt::Write as _;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use daisy_embassy::hal::{self, bind_interrupts, peripherals, usb as hal_usb};
use daisy_embassy::{led::UserLed, new_daisy_board, DaisyBoard};
use defmt::{info, unwrap, warn};
use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_time::Timer;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::Builder;
use nam_core_nostd::Engine;
use nam_hal::{ModelSource, SliceSource};
use static_cell::StaticCell;
use {defmt_rtt as _, panic_probe as _};

bind_interrupts!(struct UsbIrqs {
    OTG_FS => hal_usb::InterruptHandler<peripherals::USB_OTG_FS>;
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FRAMES: usize = daisy_embassy::audio::BLOCK_LENGTH;
#[allow(dead_code)]
const INTERLEAVED: usize = daisy_embassy::audio::HALF_DMA_BUFFER_LENGTH;
const BUDGET_CYCLES: u32 = (480_000_000 / 48_000) * FRAMES as u32;
const DITHER: f32 = 1.0e-11;
#[cfg_attr(not(feature = "qspi-model"), allow(dead_code))]
const MAX_MODEL_BYTES: usize = 16 * 1024;

const EMBEDDED_MODEL: &[u8] = include_bytes!("../model/staged.bin");

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static INPUT_GAIN: AtomicU32 = AtomicU32::new(0x3F80_0000); // 1.0f32
static PEAK_CYCLES: AtomicU32 = AtomicU32::new(0);
static CALLBACKS: AtomicU32 = AtomicU32::new(0);
static ERRORS: AtomicU32 = AtomicU32::new(0);
static CLIPPED: AtomicBool = AtomicBool::new(false);
static MODEL_ACTIVE: AtomicBool = AtomicBool::new(false);
static SELF_CYCLES: AtomicU32 = AtomicU32::new(0);
static SELF_RMS_1E6: AtomicU32 = AtomicU32::new(0);

#[cfg(feature = "tone")]
static PHASE: AtomicU32 = AtomicU32::new(0);
static ENGINE: StaticCell<Engine> = StaticCell::new();
#[cfg(feature = "qspi-model")]
#[link_section = ".sram1_bss"]
static STAGING: StaticCell<[u8; MAX_MODEL_BYTES]> = StaticCell::new();

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

const DEMCR: *mut u32 = 0xE000_EDFC as *mut u32;
const DWT_CTRL: *mut u32 = 0xE000_1000 as *mut u32;
const DWT_CYCCNT: *mut u32 = 0xE000_1004 as *mut u32;
const RCC_BASE: u32 = 0x5802_4400;
const AHB4ENR: u32 = 0xE0;
const GPIOC_BASE: u32 = 0x5802_0800;

fn dwt_init() {
    unsafe {
        core::ptr::write_volatile(DEMCR, core::ptr::read_volatile(DEMCR) | (1 << 24));
        core::ptr::write_volatile(DWT_CYCCNT, 0);
        core::ptr::write_volatile(DWT_CTRL, core::ptr::read_volatile(DWT_CTRL) | 1);
    }
}

#[inline]
fn cyccnt() -> u32 {
    unsafe { core::ptr::read_volatile(DWT_CYCCNT) }
}

fn raw_gpio_init() {
    unsafe {
        let enr = (RCC_BASE + AHB4ENR) as *mut u32;
        core::ptr::write_volatile(enr, core::ptr::read_volatile(enr) | (1 << 2));
        let moder = GPIOC_BASE as *mut u32;
        let mut m = core::ptr::read_volatile(moder);
        m &= !(0b11 << 14);
        m |= 0b01 << 14;
        core::ptr::write_volatile(moder, m);
    }
}

#[inline]
fn led_raw(on: bool) {
    unsafe {
        core::ptr::write_volatile(
            (GPIOC_BASE + 0x18) as *mut u32,
            if on { 1 << 7 } else { 1 << 23 },
        );
    }
}

fn boot_blinks(n: u32) {
    for _ in 0..n {
        led_raw(true);
        cortex_m::asm::delay(40_000_000);
        led_raw(false);
        cortex_m::asm::delay(40_000_000);
    }
}

#[inline]
fn enable_ftz() {
    unsafe {
        let mut fpscr: u32;
        core::arch::asm!("vmrs {}, fpscr", out(reg) fpscr);
        fpscr |= (1 << 24) | (1 << 25); // FZ | DN
        core::arch::asm!("vmsr fpscr, {}", in(reg) fpscr);
        // Set the same defaults for exception/ISR contexts.
        let fpdscr = 0xE000_EF3C as *mut u32;
        core::ptr::write_volatile(fpdscr, core::ptr::read_volatile(fpdscr) | (1 << 24) | (1 << 25));
    }
}

/// SAI word -> f32 (signed 24-bit in the low bits), matching zlosynth/daisy.
#[inline(always)]
fn i24_to_f32(y: u32) -> f32 {
    let y = (y.wrapping_add(0x0080_0000)) & 0x00FF_FFFF;
    (y as f32 / 8_388_608.0) - 1.0
}

#[inline(always)]
fn f32_to_i24(x: f32) -> u32 {
    if x >= 1.0 {
        CLIPPED.store(true, Ordering::Relaxed);
        return 0x7FFF_FF;
    }
    if x <= -1.0 {
        CLIPPED.store(true, Ordering::Relaxed);
        return 0x80_0000;
    }
    (x * 8_388_607.0) as i32 as u32
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

#[embassy_executor::task]
async fn heartbeat(mut led: UserLed<'static>) {
    loop {
        led.on();
        Timer::after_millis(500).await;
        led.off();
        if CLIPPED.swap(false, Ordering::Relaxed) {
            led.on();
            Timer::after_millis(60).await;
            led.off();
        }
        Timer::after_millis(500).await;
    }
}

// ---------------------------------------------------------------------------
// Audio processing
// ---------------------------------------------------------------------------

fn process_block(
    engine: &mut Option<&'static mut Engine>,
    input: &[u32],
    output: &mut [u32],
) -> u32 {
    let t0 = cyccnt();
    let gain = f32::from_bits(INPUT_GAIN.load(Ordering::Relaxed));

    let mut mono_in = [0.0f32; FRAMES];
    let mut mono_out = [0.0f32; FRAMES];
    for i in 0..FRAMES {
        mono_in[i] = i24_to_f32(input[i * 2]) * gain + DITHER;
    }

    #[cfg(feature = "tone")]
    {
        const TAU: f32 = 6.283_185_5;
        let step = TAU * 440.0 / 48_000.0;
        let mut ph = f32::from_bits(PHASE.load(Ordering::Relaxed));
        for s in mono_out.iter_mut() {
            *s = 0.25 * libm::sinf(ph);
            ph += step;
            if ph >= TAU {
                ph -= TAU;
            }
        }
        PHASE.store(ph.to_bits(), Ordering::Relaxed);
    }
    #[cfg(feature = "probe-in")]
    for i in 0..FRAMES {
        mono_out[i] = mono_in[i] * 8.0;
    }
    #[cfg(not(any(feature = "tone", feature = "probe-in")))]
    match engine {
        Some(eng) => {
            eng.process(&mono_in, &mut mono_out);
            #[cfg(feature = "boost-model")]
            for s in mono_out.iter_mut() {
                *s *= 4.0;
            }
        }
        None => mono_out.copy_from_slice(&mono_in),
    }

    for i in 0..FRAMES {
        let s = f32_to_i24(mono_out[i] - DITHER);
        output[i * 2] = s;
        output[i * 2 + 1] = s;
    }
    cyccnt().wrapping_sub(t0)
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

fn embedded_namb() -> Option<&'static [u8]> {
    if EMBEDDED_MODEL.len() <= 4 {
        return None;
    }
    let l = u32::from_le_bytes([
        EMBEDDED_MODEL[0],
        EMBEDDED_MODEL[1],
        EMBEDDED_MODEL[2],
        EMBEDDED_MODEL[3],
    ]) as usize;
    if l < 80 || EMBEDDED_MODEL.len() < 4 + l {
        return None;
    }
    Some(&EMBEDDED_MODEL[4..4 + l])
}

fn load_model_embedded(slot: &mut core::mem::MaybeUninit<Engine>) -> bool {
    match embedded_namb() {
        Some(bytes) => {
            let src = SliceSource::new(bytes);
            let bytes = src.as_slice().unwrap_or(bytes);
            Engine::init_into(bytes, slot).is_ok()
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// USB CDC logging
// ---------------------------------------------------------------------------

async fn usb_logging(usb_peripherals: daisy_embassy::usb::UsbPeripherals<'static>) {
    let mut config = hal_usb::Config::default();
    config.vbus_detection = false;

    static EP_OUT: StaticCell<[u8; 256]> = StaticCell::new();
    let ep_out = EP_OUT.init([0; 256]);
    let driver = hal_usb::Driver::new_fs(
        usb_peripherals.usb_otg_fs,
        UsbIrqs,
        usb_peripherals.pins.DP,
        usb_peripherals.pins.DN,
        ep_out,
        config,
    );

    let mut uconfig = embassy_usb::Config::new(0xc0de, 0xcafe);
    uconfig.manufacturer = Some("nam-daisy-rs");
    uconfig.product = Some("nam-pedal log");
    uconfig.device_class = 0xEF;
    uconfig.device_sub_class = 0x02;
    uconfig.device_protocol = 0x01;
    uconfig.composite_with_iads = true;

    let mut cdesc = [0u8; 256];
    let mut bdesc = [0u8; 256];
    let mut cbuf = [0u8; 64];
    let mut state = State::new();
    let mut builder = Builder::new(
        driver,
        uconfig,
        &mut cdesc,
        &mut bdesc,
        &mut [],
        &mut cbuf,
    );
    let mut class = CdcAcmClass::new(&mut builder, &mut state, 64);
    let mut usb = builder.build();

    let usb_run = usb.run();
    let logger = async {
        loop {
            class.wait_connection().await;
            let mut boot = heapless::String::<96>::new();
            let _ = write!(
                boot,
                "nam-daisy-rs model={} self_rms_1e6={} self_cycles={} budget={}\r\n",
                MODEL_ACTIVE.load(Ordering::Relaxed),
                SELF_RMS_1E6.load(Ordering::Relaxed),
                SELF_CYCLES.load(Ordering::Relaxed),
                BUDGET_CYCLES,
            );
            let _ = class.write_packet(boot.as_bytes()).await;
            loop {
                let peak = PEAK_CYCLES.swap(0, Ordering::Relaxed);
                let calls = CALLBACKS.swap(0, Ordering::Relaxed);
                let errs = ERRORS.load(Ordering::Relaxed);
                let mut line = heapless::String::<160>::new();
                let _ = write!(
                    line,
                    "peak={} ({}.{}%) calls={} err={} self={} selfpct={} model={}\r\n",
                    peak,
                    (peak as u64 * 100 / BUDGET_CYCLES as u64),
                    (peak as u64 * 1000 / BUDGET_CYCLES as u64) % 10,
                    calls,
                    errs,
                    SELF_CYCLES.load(Ordering::Relaxed),
                    (SELF_CYCLES.load(Ordering::Relaxed) as u64 * 100 / BUDGET_CYCLES as u64),
                    MODEL_ACTIVE.load(Ordering::Relaxed),
                );
                if class.write_packet(line.as_bytes()).await.is_err() {
                    break;
                }
                Timer::after_millis(1000).await;
            }
        }
    };
    join(usb_run, logger).await;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    raw_gpio_init();
    boot_blinks(3);

    info!("=== nam-daisy-rs boot ===");
    let p = hal::init(daisy_embassy::default_rcc());

    // PCM3060 (Seed 1.2 / Seed2-DFM): drive DEMP (PB11) low, as libDaisy does.
    #[cfg(feature = "seed_1_2")]
    let _demp = hal::gpio::Output::new(p.PB11, hal::gpio::Level::Low, hal::gpio::Speed::Low);

    let DaisyBoard {
        user_led,
        audio_peripherals,
        usb_peripherals,
        ..
    } = new_daisy_board!(p);

    dwt_init();
    enable_ftz();

    spawner.spawn(unwrap!(heartbeat(user_led)));

    // Build the model in place (no large stack temporary).
    let slot = ENGINE.uninit();
    let mut engine: Option<&'static mut Engine> = None;
    let loaded = load_model_embedded(slot);
    if loaded {
        let e = unsafe { slot.assume_init_mut() };
        e.stabilize();
        engine = Some(e);
        MODEL_ACTIVE.store(true, Ordering::Relaxed);
    } else {
        warn!("no valid model; starting in passthrough");
    }

    // Engine self-test: report RMS and cycle cost over USB.
    if let Some(e) = engine.as_mut() {
        let mut tin = [0.0f32; FRAMES];
        for (i, s) in tin.iter_mut().enumerate() {
            *s = 0.5 * libm::sinf(i as f32 * 0.3);
        }
        let mut tout = [0.0f32; FRAMES];
        e.process(&tin, &mut tout);
        let t0 = cyccnt();
        e.process(&tin, &mut tout);
        let dt = cyccnt().wrapping_sub(t0);
        let mut ss = 0.0f32;
        for v in tout.iter() {
            ss += v * v;
        }
        let rms = libm::sqrtf(ss / FRAMES as f32);
        SELF_RMS_1E6.store((rms * 1.0e6) as u32, Ordering::Relaxed);
        SELF_CYCLES.store(dt, Ordering::Relaxed);
        info!("self-test cycles={} rms_1e6={}", dt, SELF_RMS_1E6.load(Ordering::Relaxed));
    }

    // Audio: start the interface and enter the callback immediately (a delay
    // here overruns the SAI RX ring buffer on the master/slave PCM3060 path).
    let interface = audio_peripherals
        .prepare_interface(Default::default())
        .await;
    let mut interface = unwrap!(interface.start_interface().await);

    let audio = async {
        loop {
            let res = interface
                .start_callback(|input, output| {
                    let cycles = process_block(&mut engine, input, output);
                    let prev = PEAK_CYCLES.load(Ordering::Relaxed);
                    if cycles > prev {
                        PEAK_CYCLES.store(cycles, Ordering::Relaxed);
                    }
                    CALLBACKS.fetch_add(1, Ordering::Relaxed);
                })
                .await;
            if res.is_err() {
                ERRORS.fetch_add(1, Ordering::Relaxed);
                Timer::after_millis(10).await;
            }
        }
    };

    join(audio, usb_logging(usb_peripherals)).await;
}
