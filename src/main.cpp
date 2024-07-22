#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// TODO: Drawing a triangle > Drawing > Frames in flight

static std::vector<char> read_file(const std::string &p_path) {
  std::ifstream stream(p_path, std::ios::ate | std::ios::binary);

  if (!stream.is_open()) {
    throw std::runtime_error("failed to open file");
  }

  size_t size = static_cast<size_t>(stream.tellg());

  std::vector<char> buffer(size);

  stream.seekg(0);

  stream.read(buffer.data(), size);

  stream.close();

  return buffer;
}

// TODO: validation layers

const int g_width = 800;
const int g_height = 600;

const char *g_title = "Khronos Vulkan Tutorial";

const std::vector<const char *> g_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct queue_family_indices {
  std::optional<uint32_t> graphics;
  std::optional<uint32_t> present;

  bool is_completed() { return graphics.has_value() && present.has_value(); }
};

struct swapchain_support_details {
  VkSurfaceCapabilitiesKHR surface_capabilities;

  std::vector<VkSurfaceFormatKHR> surface_formats;

  std::vector<VkPresentModeKHR> present_modes;

  bool is_adequate() {
    return !surface_formats.empty() && !present_modes.empty();
  }
};

class application {
public:
  void run() {
    initialize_window();
    initialize_vulkan();
    execute();
    terminate_vulkan();
    terminate_window();
  }

private:
  GLFWwindow *m_window = NULL;

  VkInstance m_instance = VK_NULL_HANDLE;

  VkSurfaceKHR m_surface = VK_NULL_HANDLE;

  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;

  VkQueue m_graphics_queue = VK_NULL_HANDLE;
  VkQueue m_present_queue = VK_NULL_HANDLE;

  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  VkExtent2D m_extent = {};
  VkFormat m_format = VK_FORMAT_MAX_ENUM;

  std::vector<VkImage> m_images;
  std::vector<VkImageView> m_image_views;

  VkRenderPass m_render_pass = VK_NULL_HANDLE;

  VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
  VkPipeline m_graphics_pipeline = VK_NULL_HANDLE;

  std::vector<VkFramebuffer> m_framebuffers;

  VkCommandPool m_command_pool = VK_NULL_HANDLE;
  VkCommandBuffer m_command_buffer = VK_NULL_HANDLE;

  VkSemaphore m_image_available_semaphore = VK_NULL_HANDLE;
  VkSemaphore m_render_finished_semaphore = VK_NULL_HANDLE;
  VkFence m_in_flight_fence = VK_NULL_HANDLE;

  void initialize_window() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_window = glfwCreateWindow(g_width, g_height, g_title, NULL, NULL);
  }

  void create_instance() {
    VkApplicationInfo application_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = g_title,
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0};

    uint32_t required_instance_extension_count = 0;

    const char **required_instance_extensions =
        glfwGetRequiredInstanceExtensions(&required_instance_extension_count);

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application_info,
        .enabledExtensionCount = required_instance_extension_count,
        .ppEnabledExtensionNames = required_instance_extensions};

    if (vkCreateInstance(&create_info, NULL, &m_instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance");
    }
  }

  void create_surface() {
    if (glfwCreateWindowSurface(m_instance, m_window, NULL, &m_surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create surface");
    }
  }

  bool is_extensions_supported(VkPhysicalDevice p_physical_device) {
    uint32_t extension_count = 0;

    vkEnumerateDeviceExtensionProperties(p_physical_device, NULL,
                                         &extension_count, NULL);

    std::vector<VkExtensionProperties> extensions(extension_count);

    vkEnumerateDeviceExtensionProperties(p_physical_device, NULL,
                                         &extension_count, extensions.data());

    std::set<std::string> required_extension(g_extensions.begin(),
                                             g_extensions.end());

    for (const auto &extension : extensions) {
      required_extension.erase(extension.extensionName);
    }

    return required_extension.empty();
  }

  swapchain_support_details
  get_swapchain_support_details(VkPhysicalDevice p_physical_device) {
    swapchain_support_details swapchain_support_details = {};

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        p_physical_device, m_surface,
        &swapchain_support_details.surface_capabilities);

    uint32_t surface_format_count = 0;

    vkGetPhysicalDeviceSurfaceFormatsKHR(p_physical_device, m_surface,
                                         &surface_format_count, NULL);

    swapchain_support_details.surface_formats.resize(surface_format_count);

    vkGetPhysicalDeviceSurfaceFormatsKHR(
        p_physical_device, m_surface, &surface_format_count,
        swapchain_support_details.surface_formats.data());

    uint32_t present_mode_count = 0;

    vkGetPhysicalDeviceSurfacePresentModesKHR(p_physical_device, m_surface,
                                              &present_mode_count, NULL);

    swapchain_support_details.present_modes.resize(present_mode_count);

    vkGetPhysicalDeviceSurfacePresentModesKHR(
        p_physical_device, m_surface, &present_mode_count,
        swapchain_support_details.present_modes.data());

    return swapchain_support_details;
  }

  queue_family_indices
  get_queue_family_indices(VkPhysicalDevice p_physical_device) {
    uint32_t queue_family_count = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(p_physical_device,
                                             &queue_family_count, NULL);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);

    vkGetPhysicalDeviceQueueFamilyProperties(
        p_physical_device, &queue_family_count, queue_families.data());

    queue_family_indices queue_family_indices = {};

    uint32_t index = 0;

    for (const auto &queue_family : queue_families) {
      if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        queue_family_indices.graphics = index;
      }

      VkBool32 is_supported = VK_FALSE;

      vkGetPhysicalDeviceSurfaceSupportKHR(p_physical_device, index, m_surface,
                                           &is_supported);

      if (is_supported) {
        queue_family_indices.present = index;
      }

      if (queue_family_indices.is_completed()) {
        break;
      }

      index++;
    }

    return queue_family_indices;
  }

  bool is_physical_device_suitable(VkPhysicalDevice p_physical_device) {
    if (!is_extensions_supported(p_physical_device)) {
      return false;
    }

    swapchain_support_details swapchain_support_details =
        get_swapchain_support_details(p_physical_device);

    queue_family_indices queue_family_indices =
        get_queue_family_indices(p_physical_device);

    return swapchain_support_details.is_adequate() &&
           queue_family_indices.is_completed();
  }

  void select_physical_device() {
    uint32_t physical_device_count = 0;

    vkEnumeratePhysicalDevices(m_instance, &physical_device_count, NULL);

    if (physical_device_count == 0) {
      throw std::runtime_error("failed to find a physical device");
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);

    vkEnumeratePhysicalDevices(m_instance, &physical_device_count,
                               physical_devices.data());

    for (const auto &physical_device : physical_devices) {
      if (is_physical_device_suitable(physical_device)) {
        m_physical_device = physical_device;

        break;
      }
    }

    if (m_physical_device == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable physical device");
    }
  }

  void create_device() {
    queue_family_indices queue_family_indices =
        get_queue_family_indices(m_physical_device);

    std::set<uint32_t> indices = {queue_family_indices.graphics.value(),
                                  queue_family_indices.present.value()};

    float queue_priority = 1.0f;

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;

    for (auto index : indices) {
      VkDeviceQueueCreateInfo queue_create_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = index,
          .queueCount = 1,
          .pQueuePriorities = &queue_priority};

      queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures physical_device_features = {};

    VkDeviceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = static_cast<uint32_t>(indices.size()),
        .pQueueCreateInfos = queue_create_infos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(g_extensions.size()),
        .ppEnabledExtensionNames = g_extensions.data(),
        .pEnabledFeatures = &physical_device_features};

    if (vkCreateDevice(m_physical_device, &create_info, NULL, &m_device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create device");
    }

    vkGetDeviceQueue(m_device, queue_family_indices.graphics.value(), 0,
                     &m_graphics_queue);
    vkGetDeviceQueue(m_device, queue_family_indices.present.value(), 0,
                     &m_present_queue);
  }

  VkExtent2D select_swapchain_extent(
      const VkSurfaceCapabilitiesKHR &p_surface_capabilities) {
    if (p_surface_capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return p_surface_capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;

    glfwGetFramebufferSize(m_window, &width, &height);

    VkExtent2D extent = {static_cast<uint32_t>(width),
                         static_cast<uint32_t>(height)};

    extent.width =
        std::clamp(extent.width, p_surface_capabilities.minImageExtent.width,
                   p_surface_capabilities.maxImageExtent.width);
    extent.height =
        std::clamp(extent.height, p_surface_capabilities.minImageExtent.height,
                   p_surface_capabilities.maxImageExtent.height);

    return extent;
  }

  VkSurfaceFormatKHR select_swapchain_surface_format(
      const std::vector<VkSurfaceFormatKHR> &p_surface_formats) {
    for (const auto &surface_format : p_surface_formats) {
      if (surface_format.format == VK_FORMAT_B8G8R8_SRGB &&
          surface_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return surface_format;
      }
    }

    return p_surface_formats[0];
  }

  VkPresentModeKHR select_swapchain_present_mode(
      const std::vector<VkPresentModeKHR> &p_present_modes) {
    for (const auto &present_mode : p_present_modes) {
      if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return present_mode;
      }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
  }

  void create_swapchain() {
    swapchain_support_details swapchain_support_details =
        get_swapchain_support_details(m_physical_device);

    uint32_t min_image_count =
        swapchain_support_details.surface_capabilities.minImageCount;
    uint32_t max_image_count =
        swapchain_support_details.surface_capabilities.maxImageCount;

    uint32_t image_count = min_image_count + 1;

    if (0 < max_image_count && max_image_count < image_count) {
      image_count = max_image_count;
    }

    VkExtent2D extent =
        select_swapchain_extent(swapchain_support_details.surface_capabilities);
    VkSurfaceFormatKHR surface_format = select_swapchain_surface_format(
        swapchain_support_details.surface_formats);
    VkPresentModeKHR present_mode =
        select_swapchain_present_mode(swapchain_support_details.present_modes);

    VkSwapchainCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = m_surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform =
            swapchain_support_details.surface_capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE};

    queue_family_indices queue_family_indices =
        get_queue_family_indices(m_physical_device);

    std::vector<uint32_t> indices = {queue_family_indices.graphics.value(),
                                     queue_family_indices.present.value()};

    if (queue_family_indices.graphics != queue_family_indices.present) {
      create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = 2;
      create_info.pQueueFamilyIndices = indices.data();
    }

    if (vkCreateSwapchainKHR(m_device, &create_info, NULL, &m_swapchain) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create swapchain");
    }

    m_extent = extent;
    m_format = surface_format.format;

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count, NULL);

    m_images.resize(image_count);

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count,
                            m_images.data());
  }

  void create_image_views() {
    m_image_views.resize(m_images.size());

    for (size_t i = 0; i < m_images.size(); i++) {
      VkImageViewCreateInfo create_info = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .image = m_images[i],
          .viewType = VK_IMAGE_VIEW_TYPE_2D,
          .format = m_format,
          .components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                         .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                         .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                         .a = VK_COMPONENT_SWIZZLE_IDENTITY},
          .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .baseMipLevel = 0,
                               .levelCount = 1,
                               .baseArrayLayer = 0,
                               .layerCount = 1}};

      if (vkCreateImageView(m_device, &create_info, NULL, &m_image_views[i]) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to create image view");
      }
    };
  }

  VkShaderModule create_shader_module(const std::vector<char> &p_shader_code) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = p_shader_code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(p_shader_code.data())};

    VkShaderModule shader_module = VK_NULL_HANDLE;

    if (vkCreateShaderModule(m_device, &create_info, NULL, &shader_module) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module");
    }

    return shader_module;
  }

  void create_render_pass() {
    VkAttachmentDescription attachment_description = {
        .format = m_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};

    VkAttachmentReference attachment_reference = {
        .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass_description = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment_reference};

    VkSubpassDependency subpass_dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};

    VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &attachment_description,
        .subpassCount = 1,
        .pSubpasses = &subpass_description,
        .dependencyCount = 1,
        .pDependencies = &subpass_dependency};

    if (vkCreateRenderPass(m_device, &create_info, NULL, &m_render_pass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass");
    }
  }

  void create_graphics_pipeline() {
    auto vertex_shader_code = read_file("res/shaders/vertex.spv");
    auto fragment_shader_code = read_file("res/shaders/fragment.spv");

    auto vertex_shader_module = create_shader_module(vertex_shader_code);
    auto fragment_shader_module = create_shader_module(fragment_shader_code);

    std::vector<VkPipelineShaderStageCreateInfo> create_infos = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vertex_shader_module,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragment_shader_module,
         .pName = "main"}};

    VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE};

    VkPipelineViewportStateCreateInfo viewport_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1};

    VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE};

    VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE};

    VkPipelineColorBlendAttachmentState color_blend_attachment_state = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

    VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment_state};

    std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
                                                  VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data()};

    VkPipelineLayoutCreateInfo layout_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

    if (vkCreatePipelineLayout(m_device, &layout_create_info, NULL,
                               &m_pipeline_layout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout");
    }

    VkGraphicsPipelineCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = create_infos.data(),
        .pVertexInputState = &vertex_input_state_create_info,
        .pInputAssemblyState = &input_assembly_create_info,
        .pViewportState = &viewport_state_create_info,
        .pRasterizationState = &rasterization_state_create_info,
        .pMultisampleState = &multisample_state_create_info,
        .pColorBlendState = &color_blend_state_create_info,
        .pDynamicState = &dynamic_state_create_info,
        .layout = m_pipeline_layout,
        .renderPass = m_render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &create_info,
                                  NULL, &m_graphics_pipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline");
    }

    vkDestroyShaderModule(m_device, vertex_shader_module, NULL);
    vkDestroyShaderModule(m_device, fragment_shader_module, NULL);
  }

  void create_framebuffers() {
    m_framebuffers.resize(m_image_views.size());

    for (size_t i = 0; i < m_image_views.size(); i++) {
      VkImageView attachment = m_image_views[i];

      VkFramebufferCreateInfo create_info = {
          .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
          .renderPass = m_render_pass,
          .attachmentCount = 1,
          .pAttachments = &attachment,
          .width = m_extent.width,
          .height = m_extent.height,
          .layers = 1};

      if (vkCreateFramebuffer(m_device, &create_info, NULL,
                              &m_framebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer");
      }
    }
  }

  void create_command_pool() {
    queue_family_indices queue_family_indices =
        get_queue_family_indices(m_physical_device);

    VkCommandPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family_indices.graphics.value(),
    };

    if (vkCreateCommandPool(m_device, &create_info, NULL, &m_command_pool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool");
    }
  }

  void create_command_buffer() {
    VkCommandBufferAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    if (vkAllocateCommandBuffers(m_device, &allocate_info, &m_command_buffer) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffer");
    }
  }

  void record_command_buffer(VkCommandBuffer p_command_buffer,
                             uint32_t p_image_index) {
    VkCommandBufferBeginInfo command_buffer_begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

    if (vkBeginCommandBuffer(p_command_buffer, &command_buffer_begin_info) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to begin command buffer");
    }

    VkClearValue clear_value = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

    VkRenderPassBeginInfo render_pass_begin_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = m_render_pass,
        .framebuffer = m_framebuffers[p_image_index],
        .renderArea{.offset = {.x = 0, .y = 0}, .extent = m_extent},
        .clearValueCount = 1,
        .pClearValues = &clear_value};

    vkCmdBeginRenderPass(p_command_buffer, &render_pass_begin_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(p_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_graphics_pipeline);

    VkViewport viewport = {.x = 0.0f,
                           .y = 0.0f,
                           .width = static_cast<float>(m_extent.width),
                           .height = static_cast<float>(m_extent.height),
                           .minDepth = 0.0f,
                           .maxDepth = 1.0f};

    vkCmdSetViewport(p_command_buffer, 0, 1, &viewport);

    VkRect2D scissor = {.offset = {.x = 0, .y = 0}, .extent = m_extent};

    vkCmdSetScissor(p_command_buffer, 0, 1, &scissor);

    vkCmdDraw(p_command_buffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(p_command_buffer);

    if (vkEndCommandBuffer(p_command_buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to end command buffer");
    }
  }

  void create_synchronization_objects() {
    VkSemaphoreCreateInfo semaphore_create_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    VkFenceCreateInfo fence_create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    if (vkCreateSemaphore(m_device, &semaphore_create_info, NULL,
                          &m_image_available_semaphore) != VK_SUCCESS ||
        vkCreateSemaphore(m_device, &semaphore_create_info, NULL,
                          &m_render_finished_semaphore) != VK_SUCCESS) {
      throw std::runtime_error("failed to create semaphores");
    }

    if (vkCreateFence(m_device, &fence_create_info, NULL, &m_in_flight_fence) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create fence");
    }
  }

  void initialize_vulkan() {
    create_instance();
    create_surface();
    select_physical_device();
    create_device();
    create_swapchain();
    create_image_views();
    create_render_pass();
    create_graphics_pipeline();
    create_framebuffers();
    create_command_pool();
    create_command_buffer();
    create_synchronization_objects();
  }

  void draw_frame() {
    vkWaitForFences(m_device, 1, &m_in_flight_fence, VK_TRUE, UINT64_MAX);

    vkResetFences(m_device, 1, &m_in_flight_fence);

    uint32_t image_index = 0;

    vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                          m_image_available_semaphore, VK_NULL_HANDLE,
                          &image_index);

    vkResetCommandBuffer(m_command_buffer, 0);

    record_command_buffer(m_command_buffer, image_index);

    std::vector<VkSemaphore> wait_semaphores = {m_image_available_semaphore};

    std::vector<VkPipelineStageFlags> wait_stages = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    std::vector<VkSemaphore> signal_semaphores = {m_render_finished_semaphore};

    VkSubmitInfo submit_info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .waitSemaphoreCount = 1,
                                .pWaitSemaphores = wait_semaphores.data(),
                                .pWaitDstStageMask = wait_stages.data(),
                                .commandBufferCount = 1,
                                .pCommandBuffers = &m_command_buffer,
                                .signalSemaphoreCount = 1,
                                .pSignalSemaphores = signal_semaphores.data()};

    if (vkQueueSubmit(m_graphics_queue, 1, &submit_info, m_in_flight_fence) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer");
    }

    std::vector<VkSwapchainKHR> swapchains = {m_swapchain};

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = signal_semaphores.data(),
        .swapchainCount = 1,
        .pSwapchains = swapchains.data(),
        .pImageIndices = &image_index};

    vkQueuePresentKHR(m_present_queue, &present_info);
  }

  void execute() {
    while (!glfwWindowShouldClose(m_window)) {
      glfwPollEvents();

      draw_frame();
    }
  }

  void terminate_vulkan() {
    vkDestroySemaphore(m_device, m_image_available_semaphore, NULL);
    vkDestroySemaphore(m_device, m_render_finished_semaphore, NULL);
    vkDestroyFence(m_device, m_in_flight_fence, NULL);

    vkDestroyCommandPool(m_device, m_command_pool, NULL);

    for (auto &framebuffer : m_framebuffers) {
      vkDestroyFramebuffer(m_device, framebuffer, NULL);
    }

    vkDestroyPipeline(m_device, m_graphics_pipeline, NULL);
    vkDestroyPipelineLayout(m_device, m_pipeline_layout, NULL);
    vkDestroyRenderPass(m_device, m_render_pass, NULL);

    for (auto &image_view : m_image_views) {
      vkDestroyImageView(m_device, image_view, NULL);
    }

    vkDestroySwapchainKHR(m_device, m_swapchain, NULL);
    vkDestroyDevice(m_device, NULL);
    vkDestroySurfaceKHR(m_instance, m_surface, NULL);
    vkDestroyInstance(m_instance, NULL);
  }

  void terminate_window() {
    glfwDestroyWindow(m_window);

    glfwTerminate();
  }
};

int main() {
  application application;

  try {
    application.run();
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << std::endl;

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}